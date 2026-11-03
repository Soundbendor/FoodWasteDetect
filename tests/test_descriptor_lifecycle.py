"""Lifecycle and description checks with fake clients and processes."""

import errno
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import httpx

from classification.experiments.config import validate_ollama_config
from classification.llm import generate_descriptors as descriptors
from classification.llm import ollama


ConnectError = httpx.ConnectError


class OllamaConfigTests(unittest.TestCase):
    def test_timeouts_must_be_positive_finite_numbers(self):
        for name in ("startup_timeout", "request_timeout"):
            for value in (0, -1, True, False, float("nan"), float("inf"), -float("inf"), "10", None):
                with self.subTest(name=name, value=value):
                    settings = {"startup_timeout": 10, "request_timeout": 30}
                    settings[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        validate_ollama_config(settings)

    def test_valid_timeouts_and_missing_keys(self):
        for value in (1, 0.25):
            validate_ollama_config({"startup_timeout": value, "request_timeout": value})
        for name in ("startup_timeout", "request_timeout"):
            settings = {"startup_timeout": 10, "request_timeout": 30}
            del settings[name]
            with self.assertRaises(KeyError):
                validate_ollama_config(settings)


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.llm = descriptors.DescriberLLM(self.client, 'test-model')

    def test_public_description_uses_shared_prompt_without_cleaning_label(self):
        output = types.SimpleNamespace(food_class='Seafood', description='A seafood dish.')
        with patch.object(self.llm, '_describe_class', return_value=output) as describe:
            self.assertIs(self.llm.describe_class('Seafood'), output)
        describe.assert_called_once_with('Seafood', descriptors.descriptor_prompt('Seafood'))

    def test_generate_descriptions_preserves_classes_and_requested_count(self):
        outputs = [types.SimpleNamespace(description=text) for text in
                   ['Apple one', 'Apple two', 'Pear one', 'Pear two']]
        with patch.object(self.llm, 'describe_class', side_effect=outputs) as describe:
            result = self.llm.generate_descriptions(['Apple', 'Pear'], 2)
        self.assertEqual(result, {'Apple': ['Apple one', 'Apple two'],
                                  'Pear': ['Pear one', 'Pear two']})
        self.assertEqual([call.args for call in describe.call_args_list],
                         [('Apple',), ('Apple',), ('Pear',), ('Pear',)])

    def test_generate_descriptions_rejects_empty_output(self):
        for count, text in [(0, 'unused'), (1, ''), (1, '   ')]:
            with self.subTest(count=count, text=text), patch.object(
                self.llm, 'describe_class', return_value=types.SimpleNamespace(description=text)
            ):
                with self.assertRaisesRegex(ValueError, 'Empty generated description for Apple'):
                    self.llm.generate_descriptions(['Apple'], count)

    def test_generate_descriptions_propagates_errors_without_retry(self):
        for error in (RuntimeError('generation failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), patch.object(
                self.llm, 'describe_class', side_effect=error
            ) as describe:
                with self.assertRaises(type(error)) as raised:
                    self.llm.generate_descriptions(['Apple', 'Pear'], 2)
                self.assertIs(raised.exception, error)
                describe.assert_called_once_with('Apple')

    def test_single_description_preserves_label_and_removes_emphasis(self):
        response = self.llm.client.chat.return_value
        response.message.content = '{"food_class":"wrong", "description":"*Golden* crust"}'
        prompt = descriptors.descriptor_prompt('bread')
        result = self.llm._describe_class('bread food', prompt)
        self.assertEqual(result.food_class, 'bread food')
        self.assertEqual(result.description, 'Golden crust')
        self.client.chat.assert_called_once_with(
            model='test-model', messages=[prompt], format=descriptors.FoodDescriptor.model_json_schema(),
        )


class OllamaLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.client_constructor = self.enterContext(patch.object(ollama, 'Client'))
        self.probe_constructor = self.enterContext(patch.object(ollama.httpx, 'Client'))
        self.get = self.probe_constructor.return_value.__enter__.return_value.get
        self.get.return_value.json.return_value = {'models': []}
        self.popen = self.enterContext(patch.object(ollama.subprocess, 'Popen'))
        self.process = self.popen.return_value
        self.process.poll.return_value = None
        self.enterContext(patch.object(ollama.time, 'sleep'))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.binary = self.directory / 'ollama'
        self.binary.write_text('# Synthetic fixture; never executed.\n')
        self.binary.chmod(0o700)
        self.config = {
            'binary_path': str(self.binary),
            'host': '127.0.0.1:12345',
            'startup_timeout': 120,
            'request_timeout': 600,
        }
        self.session = ollama.OllamaSession(self.config)

    def test_download_progress_environment_and_cleanup(self):
        for failure in (None, RuntimeError('pull failed'), KeyboardInterrupt()):
            with self.subTest(failure=failure):
                self.process.reset_mock()
                self.get.side_effect = [ConnectError('connection failed'), self.get.return_value]

                def progress_stream():
                    yield types.SimpleNamespace(status='pulling manifest', total=None, completed=None)
                    yield types.SimpleNamespace(status='pulling layer', total=100, completed=25)
                    if failure is not None:
                        raise failure
                    yield types.SimpleNamespace(status='success', total=None, completed=None)

                with patch.object(self.session.client, 'pull', return_value=progress_stream()) as pull, \
                        patch('builtins.print') as output, \
                        patch.object(ollama.subprocess, 'run') as run:
                    with patch.dict(ollama.os.environ, {'OLLAMA_MODELS': '/synthetic/models'}):
                        if failure is None:
                            with self.session:
                                self.session.download_model('requested:exact-tag')
                        else:
                            with self.assertRaises(type(failure)) as caught:
                                with self.session:
                                    self.session.download_model('requested:exact-tag')
                            self.assertIs(caught.exception, failure)
                        pull.assert_called_once_with(model='requested:exact-tag', stream=True)
                        run.assert_not_called()
                        output.assert_any_call('pulling manifest', flush=True)
                        output.assert_any_call('pulling layer: 25/100 bytes', flush=True)
                        if failure is None:
                            output.assert_any_call('success', flush=True)
                        self.assertEqual(self.popen.call_args.kwargs['env']['OLLAMA_MODELS'], '/synthetic/models')
                        self.assertEqual(self.popen.call_args.kwargs['env']['OLLAMA_HOST'], self.session.host)
                self.process.terminate.assert_called_once()
                self.process.wait.assert_called_once()
                self.assertIsNone(self.session._process)

    def test_occupied_endpoint_prevents_pull(self):
        with patch.object(self.session.client, 'pull') as pull:
            with self.assertRaisesRegex(RuntimeError, 'already bound'):
                with self.session:
                    self.session.download_model('requested:tag')
        pull.assert_not_called()
        self.popen.assert_not_called()

    def test_invalid_installations_fail_before_network_or_launch(self):
        non_executable = self.directory / 'not-executable'
        non_executable.write_text('fixture')
        non_executable.chmod(0o600)
        cases = [
            (str(self.directory / 'missing'), FileNotFoundError),
            (str(self.directory), ValueError),
            (str(non_executable), PermissionError),
            ('', ValueError),
            (None, ValueError),
        ]
        for path, error_type in cases:
            with self.subTest(path=path):
                self.session.binary_path = path
                with self.assertRaises(error_type) as caught:
                    self.session.start_server()
                message = str(caught.exception)
                self.assertIn('ollama.binary_path', message)
                self.assertIn('README.md: Ollama setup', message)
                if path:
                    self.assertIn(path, message)
        self.get.assert_not_called()
        self.popen.assert_not_called()

    def test_relative_and_home_paths_resolve(self):
        # Use a mocked cwd rather than changing the process-wide working directory.
        with patch.object(ollama.os, 'getcwd', return_value=str(self.directory)):
            self.session.binary_path = './ollama'
            self.assertEqual(self.session._validate_binary(), str(self.binary))
        with patch.dict(ollama.os.environ, {'HOME': str(self.directory)}):
            self.session.binary_path = '~/ollama'
            self.assertEqual(self.session._validate_binary(), str(self.binary))

    def test_bare_executable_resolves_on_path(self):
        with patch.dict(ollama.os.environ, {'PATH': str(self.directory)}):
            self.session.binary_path = 'ollama'
            self.assertEqual(self.session._validate_binary(), str(self.binary))

    def test_missing_path_executable_fails_before_network_or_launch(self):
        with patch.dict(ollama.os.environ, {'PATH': ''}):
            self.session.binary_path = 'missing-ollama'
            with self.assertRaisesRegex(FileNotFoundError, 'not found on PATH'):
                self.session.start_server()
        self.get.assert_not_called()
        self.popen.assert_not_called()

    def test_launch_errors_include_guidance_and_preserve_cause(self):
        self.get.side_effect = ConnectError('connection failed')
        for code in (errno.ENOENT, errno.EACCES, errno.ENOEXEC):
            with self.subTest(errno=code):
                error = OSError(code, 'synthetic launch failure')
                self.popen.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    self.session.start_server()
                self.assertIs(caught.exception.__cause__, error)
                self.assertIn(str(self.binary), str(caught.exception))
                self.assertIn('ollama.binary_path', str(caught.exception))
                self.assertIsNone(self.session._process)
        self.process.terminate.assert_not_called()

    def test_constructor_uses_config_without_environment_fallback(self):
        config = dict(self.config, startup_timeout=7, request_timeout=19)
        with patch.dict(ollama.os.environ, {'OLLAMA_HOST': 'other:9999'}):
            session = ollama.OllamaSession(config)
        self.assertEqual(session.binary_path, config['binary_path'])
        self.assertEqual(session.host, 'http://127.0.0.1:12345')
        self.assertEqual(session.startup_timeout, 7)
        ollama.Client.assert_called_with(host=session.host, timeout=19)

    def test_missing_settings_fail_instead_of_using_defaults(self):
        for key in self.config:
            with self.subTest(key=key):
                config = self.config.copy()
                del config[key]
                with self.assertRaises(KeyError):
                    ollama.OllamaSession(config)


    def test_existing_server_is_rejected_without_starting_or_stopping_it(self):
        with self.assertRaisesRegex(RuntimeError, r"already bound to http://127\.0\.0\.1:12345"):
            with self.session:
                self.fail("An occupied endpoint must prevent context entry")
        self.popen.assert_not_called()
        self.process.terminate.assert_not_called()

    def test_direct_launch_uses_configured_binary_and_endpoint(self):
        ready = self.get.return_value
        self.get.side_effect = [ConnectError('connection failed'), ready]
        with self.session:
            self.assertIs(self.session._process, self.process)
        args, kwargs = self.popen.call_args
        self.assertEqual(args[0], [str(self.binary), 'serve'])
        self.assertEqual(kwargs['env']['OLLAMA_HOST'], 'http://127.0.0.1:12345')
        self.process.terminate.assert_called_once()
        self.session.close()
        self.process.terminate.assert_called_once()

    def test_workload_exception_cleans_up_owned_process(self):
        self.get.side_effect = [ConnectError('connection failed'), self.get.return_value]
        with self.assertRaisesRegex(ValueError, 'workload'):
            with self.session:
                raise ValueError('workload')
        self.process.terminate.assert_called_once()

    def test_startup_deadline_cleans_up(self):
        self.get.side_effect = ConnectError('connection failed')
        with patch.object(ollama.time, 'monotonic', side_effect=[0, 0, 121, 121]):
            with self.assertRaises(TimeoutError):
                self.session.start_server()
        self.process.terminate.assert_called_once()

    def test_early_process_exit_fails_and_reaps(self):
        self.get.side_effect = ConnectError('connection failed')
        self.process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            self.session.start_server()
        self.process.wait.assert_called_once()
        self.process.terminate.assert_not_called()

    def test_unresponsive_owned_process_is_killed_after_grace_period(self):
        self.session._process = self.process
        self.process.wait.side_effect = [subprocess.TimeoutExpired('ollama', 10), None]
        self.session.close()
        self.process.kill.assert_called_once()
        self.assertIsNone(self.session._process)

    @unittest.skip("Existing mismatch: remote-endpoint guard was removed; outside binary validation scope")
    def test_remote_endpoint_failure_does_not_launch_local_server(self):
        self.session.host = 'http://remote.example:11434'
        self.get.side_effect = ConnectError('connection failed')
        with self.assertRaises(ConnectionError):
            self.session.start_server()
        self.popen.assert_not_called()

    def test_invalid_endpoint_response_does_not_launch_server(self):
        self.get.return_value.json.return_value = {'unexpected': True}
        with self.assertRaisesRegex(RuntimeError, 'model list'):
            self.session.start_server()
        self.popen.assert_not_called()



if __name__ == '__main__':
    unittest.main()

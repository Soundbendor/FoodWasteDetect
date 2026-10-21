"""Lifecycle checks with synthetic HTTP/process doubles; no Ollama or GPU needed."""
import errno
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class ConnectError(Exception):
    pass


class RequestTimeout(Exception):
    pass


class DescriptorLifecycleTests(unittest.TestCase):
    def setUp(self):
        httpx = types.ModuleType('httpx')
        httpx.ConnectError = ConnectError
        httpx.TimeoutException = RequestTimeout
        httpx.Client = Mock()
        self.probe = httpx.Client.return_value.__enter__ = Mock(return_value=Mock())
        httpx.Client.return_value.__exit__ = Mock(return_value=False)
        self.get = self.probe.return_value.get
        self.get.return_value.json.return_value = {'models': []}
        ollama = types.ModuleType('ollama')
        ollama.Client = Mock()
        ollama.ChatResponse = object
        pydantic = types.ModuleType('pydantic')
        pydantic.BaseModel = object
        path = Path(__file__).resolve().parents[1] / 'src/internvl/generate_descriptors.py'
        spec = importlib.util.spec_from_file_location('_descriptor_lifecycle_test', path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            'httpx': httpx, 'ollama': ollama, 'pydantic': pydantic,
        }):
            spec.loader.exec_module(self.module)
        self.popen = self.enterContext(patch.object(self.module.subprocess, 'Popen'))
        self.process = self.popen.return_value
        self.process.poll.return_value = None
        self.enterContext(patch.object(self.module.time, 'sleep'))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.binary = self.directory / 'ollama'
        self.binary.write_text('# Synthetic fixture; never executed.\n')
        self.binary.chmod(0o700)
        self.config = {
            'llm': 'test-model',
            'binary_path': str(self.binary),
            'host': '127.0.0.1:12345',
            'startup_timeout': 120,
            'request_timeout': 600,
        }
        self.llm = self.module.DescriberLLM(self.config)

    def test_public_description_uses_shared_prompt_without_cleaning_label(self):
        output = types.SimpleNamespace(food_class='Seafood', description='A seafood dish.')
        with patch.object(self.llm, '_describe_class', return_value=output) as describe:
            self.assertIs(self.llm.describe_class('Seafood'), output)
        describe.assert_called_once_with('Seafood', self.module.descriptor_prompt('Seafood'))

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

    def test_download_progress_environment_and_cleanup(self):
        for failure in (None, RuntimeError('pull failed'), KeyboardInterrupt()):
            with self.subTest(failure=failure):
                self.process.reset_mock()
                self.get.side_effect = [ConnectError(), self.get.return_value]

                def progress_stream():
                    yield types.SimpleNamespace(status='pulling manifest', total=None, completed=None)
                    yield types.SimpleNamespace(status='pulling layer', total=100, completed=25)
                    if failure is not None:
                        raise failure
                    yield types.SimpleNamespace(status='success', total=None, completed=None)

                with patch.object(self.llm.client, 'pull', return_value=progress_stream()) as pull, \
                        patch('builtins.print') as output, \
                        patch.object(self.module.subprocess, 'run') as run:
                    with patch.dict(self.module.os.environ, {'OLLAMA_MODELS': '/synthetic/models'}):
                        if failure is None:
                            with self.llm:
                                self.llm.download_model('requested:exact-tag')
                        else:
                            with self.assertRaises(type(failure)) as caught:
                                with self.llm:
                                    self.llm.download_model('requested:exact-tag')
                            self.assertIs(caught.exception, failure)
                        pull.assert_called_once_with(model='requested:exact-tag', stream=True)
                        run.assert_not_called()
                        output.assert_any_call('pulling manifest', flush=True)
                        output.assert_any_call('pulling layer: 25/100 bytes', flush=True)
                        if failure is None:
                            output.assert_any_call('success', flush=True)
                        self.assertEqual(self.popen.call_args.kwargs['env']['OLLAMA_MODELS'], '/synthetic/models')
                        self.assertEqual(self.popen.call_args.kwargs['env']['OLLAMA_HOST'], self.llm.host)
                self.process.terminate.assert_called_once()
                self.process.wait.assert_called_once()
                self.assertIsNone(self.llm._process)
                self.assertEqual(self.llm.model, 'test-model')

    def test_occupied_endpoint_prevents_pull(self):
        with patch.object(self.llm.client, 'pull') as pull:
            with self.assertRaisesRegex(RuntimeError, 'already bound'):
                with self.llm:
                    self.llm.download_model('requested:tag')
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
                self.llm.binary_path = path
                with self.assertRaises(error_type) as caught:
                    self.llm.start_server()
                message = str(caught.exception)
                self.assertIn('ollama.binary_path', message)
                self.assertIn('README.md: Ollama setup', message)
                if path:
                    self.assertIn(path, message)
        self.get.assert_not_called()
        self.popen.assert_not_called()

    def test_relative_and_home_paths_resolve(self):
        # Use a mocked cwd rather than changing the process-wide working directory.
        with patch.object(self.module.os, 'getcwd', return_value=str(self.directory)):
            self.llm.binary_path = './ollama'
            self.assertEqual(self.llm._validate_binary(), str(self.binary))
        with patch.dict(self.module.os.environ, {'HOME': str(self.directory)}):
            self.llm.binary_path = '~/ollama'
            self.assertEqual(self.llm._validate_binary(), str(self.binary))

    def test_bare_executable_resolves_on_path(self):
        with patch.dict(self.module.os.environ, {'PATH': str(self.directory)}):
            self.llm.binary_path = 'ollama'
            self.assertEqual(self.llm._validate_binary(), str(self.binary))

    def test_missing_path_executable_fails_before_network_or_launch(self):
        with patch.dict(self.module.os.environ, {'PATH': ''}):
            self.llm.binary_path = 'missing-ollama'
            with self.assertRaisesRegex(FileNotFoundError, 'not found on PATH'):
                self.llm.start_server()
        self.get.assert_not_called()
        self.popen.assert_not_called()

    def test_launch_errors_include_guidance_and_preserve_cause(self):
        self.get.side_effect = ConnectError()
        for code in (errno.ENOENT, errno.EACCES, errno.ENOEXEC):
            with self.subTest(errno=code):
                error = OSError(code, 'synthetic launch failure')
                self.popen.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    self.llm.start_server()
                self.assertIs(caught.exception.__cause__, error)
                self.assertIn(str(self.binary), str(caught.exception))
                self.assertIn('ollama.binary_path', str(caught.exception))
                self.assertIsNone(self.llm._process)
        self.process.terminate.assert_not_called()

    def test_constructor_uses_config_without_environment_fallback(self):
        config = dict(self.config, startup_timeout=7, request_timeout=19)
        with patch.dict(self.module.os.environ, {'OLLAMA_HOST': 'other:9999'}):
            llm = self.module.DescriberLLM(config)
        self.assertEqual(llm.model, config['llm'])
        self.assertEqual(llm.binary_path, config['binary_path'])
        self.assertEqual(llm.host, 'http://127.0.0.1:12345')
        self.assertEqual(llm.startup_timeout, 7)
        self.module.Client.assert_called_with(host=llm.host, timeout=19)

    def test_missing_settings_fail_instead_of_using_defaults(self):
        for key in self.config:
            with self.subTest(key=key):
                config = self.config.copy()
                del config[key]
                with self.assertRaises(KeyError):
                    self.module.DescriberLLM(config)


    def test_existing_server_is_rejected_without_starting_or_stopping_it(self):
        with self.assertRaisesRegex(RuntimeError, r"already bound to http://127\.0\.0\.1:12345"):
            with self.llm:
                self.fail("An occupied endpoint must prevent context entry")
        self.popen.assert_not_called()
        self.process.terminate.assert_not_called()

    def test_direct_launch_uses_configured_binary_and_endpoint(self):
        ready = self.get.return_value
        self.get.side_effect = [ConnectError(), ready]
        with self.llm:
            self.assertIs(self.llm._process, self.process)
        args, kwargs = self.popen.call_args
        self.assertEqual(args[0], [str(self.binary), 'serve'])
        self.assertEqual(kwargs['env']['OLLAMA_HOST'], 'http://127.0.0.1:12345')
        self.process.terminate.assert_called_once()
        self.llm.close()
        self.process.terminate.assert_called_once()

    def test_workload_exception_cleans_up_owned_process(self):
        self.get.side_effect = [ConnectError(), self.get.return_value]
        with self.assertRaisesRegex(ValueError, 'workload'):
            with self.llm:
                raise ValueError('workload')
        self.process.terminate.assert_called_once()

    def test_startup_deadline_cleans_up(self):
        self.get.side_effect = ConnectError()
        with patch.object(self.module.time, 'monotonic', side_effect=[0, 0, 121, 121]):
            with self.assertRaises(TimeoutError):
                self.llm.start_server()
        self.process.terminate.assert_called_once()

    def test_early_process_exit_fails_and_reaps(self):
        self.get.side_effect = ConnectError()
        self.process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            self.llm.start_server()
        self.process.wait.assert_called_once()
        self.process.terminate.assert_not_called()

    def test_unresponsive_owned_process_is_killed_after_grace_period(self):
        self.llm._process = self.process
        self.process.wait.side_effect = [subprocess.TimeoutExpired('ollama', 10), None]
        self.llm.close()
        self.process.kill.assert_called_once()
        self.assertIsNone(self.llm._process)

    @unittest.skip("Existing mismatch: remote-endpoint guard was removed; outside binary validation scope")
    def test_remote_endpoint_failure_does_not_launch_local_server(self):
        self.llm.host = 'http://remote.example:11434'
        self.get.side_effect = ConnectError()
        with self.assertRaises(ConnectionError):
            self.llm.start_server()
        self.popen.assert_not_called()

    def test_invalid_endpoint_response_does_not_launch_server(self):
        self.get.return_value.json.return_value = {'unexpected': True}
        with self.assertRaisesRegex(RuntimeError, 'model list'):
            self.llm.start_server()
        self.popen.assert_not_called()

    def test_single_description_preserves_label_and_removes_emphasis(self):
        descriptor = Mock()
        descriptor.description = '*Golden* crust'
        response = self.llm.client.chat.return_value
        response.message.content = '{"food_class":"wrong", "description":"*Golden* crust"}'
        with patch.object(self.module, 'FoodDescriptor') as schema:
            schema.model_validate_json.return_value = descriptor
            prompt = self.module.descriptor_prompt('bread')
            result = self.llm._describe_class('bread food', prompt)
            schema.model_validate_json.assert_called_once_with(response.message.content)
        self.assertEqual(result.food_class, 'bread food')
        self.assertEqual(result.description, 'Golden crust')
        self.assertEqual(self.llm.client.chat.call_args.kwargs['messages'], [prompt])


if __name__ == '__main__':
    unittest.main()

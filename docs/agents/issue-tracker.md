# Issue tracker: GitHub

Issues and specs live in Soundbendor/FoodWasteDetect on GitHub.
Use the `gh` CLI with `--repo Soundbendor/FoodWasteDetect`.

## Operations

- Create: `gh issue create --title "..." --body-file <file>`
- Read: `gh issue view <number> --json number,title,body,labels,comments`
- List: `gh issue list --state open --json number,title,body,labels`
- Comment: `gh issue comment <number> --body-file <file>`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: post the explanation as a comment, then `gh issue close <number>`.

Include the explicit `--repo` argument on each command. Write multiline
bodies to a temporary file and pass `--body-file`.

Before the first issue creation in a session, verify the target with
`gh repo view --json nameWithOwner`. Resolve any mismatch before publishing.

## Relationships

Use native GitHub sub-issues and blocking dependencies when supported.
Check the installed `gh` command's help before using relationship flags.

If native relationships are unavailable, put `Part of #<parent>` or
`Blocked by: #<number>, #<number>` at the top of the child issue body.
Include child issues in a task list on the parent. An issue is unblocked
when every blocker is closed.

## Skill routing

"Publish to the issue tracker" means create a GitHub issue.
"Fetch the relevant ticket" means read the GitHub issue, including comments.

## Pull requests as a triage surface

PRs as a request surface: no.

# AI metadata review

Optional review for `salmon up`, `salmon tag` and their web jobs. Checks album metadata and release type against release pages.
Does not inspect audio or edit individual track titles or filenames. Disabled by default.

Release types use RED's categories for both RED and OPS; the review explains its choice.
New genres need named citations from opened pages. Unsupported removals keep existing tags; likely typos get a warning.

## Use Codex with your ChatGPT plan

Install a recent [Codex CLI](https://developers.openai.com/codex/cli), then run `codex login` and sign in with
ChatGPT. Codex must be on salmon's `PATH` and logged in as the user running salmon.

Add to `config.toml`:

```toml
[upload.ai_review]
enabled = true
provider = "codex"
codex_model = "gpt-6.1-sol"
reasoning_effort = "high"
use_web_search = true
timeout_seconds = 300
```

No API key needed. Your Codex subscription limits apply. Start with high reasoning; use medium for faster reviews.
Each review runs in a temporary, read-only session with shell and plugins disabled. Your Codex settings are ignored;
its saved login is reused. API users can keep their existing settings: `provider` defaults to `"openai"`.

## Review flow

Manual review → AI suggestions → apply or keep original → manual review after applying.
You can also give the model instructions and rerun. Codex reruns start a fresh session.

- `--skip-initial-review`: AI first.
- `--apply-ai-suggestions` or `-yyy`: automatically run and apply AI suggestions.

If AI review fails, salmon falls back to normal metadata review. All settings are in
[config.reference.toml](../src/salmon/data/config.reference.toml).

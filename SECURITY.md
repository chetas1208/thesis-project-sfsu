# Security policy

This is a research repository that can call hosted model APIs and local model services. Treat credentials and source corpora as sensitive.

## Do not commit

- API keys, access tokens, passwords, private keys, or bearer tokens.
- `.env` files or copied shell histories.
- Personal resumes, scraped private records, or derived files containing personal data.
- Model caches, checkpoints, adapters, or runtime logs that may contain prompts or environment details.

Use the checked-in `.env.example` templates and provide values through the local environment. The root `.gitignore` excludes credential files and local/generated state.

## If a secret is exposed

1. Revoke or rotate the credential immediately at its provider.
2. Stop using the affected token, even if the file was only pushed briefly.
3. Remove the secret from the working tree and Git history using an approved history-rewrite procedure.
4. If it reached GitHub, use GitHub’s secret-revocation guidance and open a private security report rather than posting the value publicly.

Do not include the secret in an issue, pull request, commit message, or test fixture. Report only the affected path, provider, and approximate exposure time.

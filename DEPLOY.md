# Deploying online

The online app is three free pieces plus the HPC:

| Piece | Where | Holds |
| --- | --- | --- |
| Web app | GitHub Pages | The built React frontend, no secrets |
| API | Hugging Face Space (Docker, free CPU) | FastAPI backend, all server secrets |
| Data and sign-in | Supabase | Postgres, a private Storage bucket, Auth |
| Primary LLM | IITD HPC, through your Mac | vLLM behind an API key, reached via Tailscale Funnel |

Visitors get an anonymous account on first load, so their project is saved on
the server without any sign-up. It stays until they press **New project**,
which deletes their datasets, models and chat. "Save this project" in the
header attaches an email or Google login to the same account so they can come
back on another device.

When the HPC is not being served, the API falls back to the cloud keys
(OpenRouter, Gemini, Groq, Cerebras) within a few seconds and keeps using them
until the HPC answers again.

Local development is unchanged: `./run.sh` needs none of this.

Placeholders used below: `<user>` is your GitHub user name, `<repo>` the GitHub
repository, `<hf-user>/<space>` the Hugging Face Space, `<ref>` the Supabase
project reference.

---

## 1. GitHub repository

The workflows deploy from `main`.

```bash
cd proj2
git status                  # nothing under random/, .env, data_storage/ or hpc/.llm_api_key
git branch -M main
git add -A && git commit -m "Initial commit"
gh repo create <repo> --private --source . --push
```

`.gitignore` already keeps secrets, datasets, the local database and
`random/` out. Look over the untracked files at the root before the first
commit (`.cursorrules`, `Project_Resume_Highlights.*`, `make_pitch_slide.py`,
`test_time.csv`, ...) and delete or ignore any you do not want published.

GitHub Pages on a private repository needs GitHub Pro, which you have.

## 2. Supabase

Create a project (any region close to India, e.g. Mumbai), then:

1. **Database URL.** *Connect* → *Session pooler* → copy the URI:
   `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`.
   URL-encode the password if it has symbols. This is `DATABASE_URL`. The
   backend switches it to asyncpg itself; do not add `sslmode`. The *Direct
   connection* string is IPv6-only and will not work from Hugging Face.
2. **Bucket and lockdown.** *SQL Editor* → paste `supabase/setup.sql` → *Run*.
   It creates the private `sst-data` bucket. The backend creates its tables and
   enables row-level security on them at its first start; run the script once
   more afterwards and check every row reads `rls_enabled = true`, `policies = 0`.
3. **Auth.** *Authentication* →
   - *Sign In / Providers*: turn on **Allow anonymous sign-ins** and
     **Allow manual linking** (needed to attach Google to an anonymous user).
     Email stays on.
   - *URL Configuration*: Site URL `https://<user>.github.io/<repo>/`; add the
     redirect URLs `https://<user>.github.io/<repo>/**` and
     `http://localhost:5173/**`.
   - *Emails → SMTP*: the built-in sender allows only a few emails an hour.
     Add your own SMTP (e.g. Resend, Brevo) before sharing the app widely.
   - Optional: *Attack Protection* → enable CAPTCHA if anonymous sign-ups get
     abused (the frontend would then need the CAPTCHA token; not wired yet).
4. **Google (optional).** Create an OAuth client in Google Cloud (Web
   application, authorised redirect URI
   `https://<ref>.supabase.co/auth/v1/callback`) and paste its id and secret
   into the Google provider.
5. **Keys.** *Project Settings* → *API Keys*:
   - the **publishable** key (`sb_publishable_...`, or the legacy `anon` key)
     goes to the frontend;
   - the **secret** key (`sb_secret_...`, or the legacy `service_role` key)
     goes only into the Space secrets. It bypasses row-level security.

   The backend verifies user tokens against the project's JWKS, which works
   with the default asymmetric signing keys. Only if the project still signs
   with the legacy shared secret, also set `SUPABASE_JWT_SECRET` (*JWT Keys*
   page).

## 3. Tailscale Funnel (public URL for the HPC model)

The Funnel gives the Mac's forwarded vLLM port a public HTTPS address. Only
the `/v1` path is published, and vLLM rejects any `/v1` request without the
API key.

1. Install and log in: `brew install --cask tailscale` (or the App Store
   app), open it, sign in.
2. In the [admin console](https://login.tailscale.com/admin): *DNS* → enable
   MagicDNS and **HTTPS Certificates**; *Access controls* → add the Funnel
   attribute:
   ```json
   "nodeAttrs": [{ "target": ["autogroup:member"], "attr": ["funnel"] }]
   ```
   (Running `tailscale funnel 8888` once also prints a link that does this.)
3. Serve the model:
   ```bash
   ssh iitd                    # authenticate once; leave it open
   ./run.sh --serve-llm
   ```
   The first run creates the API key on the cluster (`~/mtp/hpc/.llm_api_key`,
   readable only by you) and copies it into `.env` as `LOCAL_LLM_API_KEY`.
   If a job started before the key existed is still running, `run.sh` refuses
   to publish it; stop it with `ssh iitd qdel <job id>` and run again.
4. `run.sh` prints the public URL, `https://<mac>.<tailnet>.ts.net/v1`. It
   stays the same as long as the machine name does. This is
   `LOCAL_LLM_BASE_URL`.

## 4. Hugging Face Space (API)

1. Create a Space: SDK **Docker**, blank template, hardware *CPU basic*
   (free), visibility **Public**. A private Space cannot be called from the
   browser. Its files are public, which is fine: they contain no secrets.
2. *Settings* → *Variables and secrets*:

   | Name | Kind | Value |
   | --- | --- | --- |
   | `DATABASE_URL` | secret | Supabase session pooler URI |
   | `SUPABASE_URL` | variable | `https://<ref>.supabase.co` |
   | `SUPABASE_SERVICE_ROLE_KEY` | secret | Supabase secret key |
   | `CORS_ALLOW_ORIGINS` | variable | `https://<user>.github.io` |
   | `LOCAL_LLM_BASE_URL` | secret | Funnel URL ending in `/v1` |
   | `LOCAL_LLM_API_KEY` | secret | `LOCAL_LLM_API_KEY` from your `.env` |
   | `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY` | secret | whichever you have; used when the HPC is off |

   Copy the vLLM key into the form without printing it, e.g.
   `grep '^LOCAL_LLM_API_KEY=' .env | cut -d= -f2- | pbcopy`.

   The image already sets `AUTH_REQUIRED=true`, `STORAGE_BACKEND=supabase` and
   `LLM_MODE=local_first`. Optional limits (defaults in brackets):
   `MAX_UPLOAD_MB` (25), `CHAT_RATE_LIMIT_PER_HOUR` (40 per user),
   `MAX_DATASETS_PER_USER` (3; uploading a fourth deletes the oldest),
   `MAX_PROJECT_STATE_MB` (5).
3. *Settings* → *Access Tokens* on your HF profile → a fine-grained token with
   write access to this Space only.
4. The API URL is `https://<hf-user>-<space>.hf.space` (dots and underscores
   in the names become dashes).

## 5. GitHub settings

*Settings* → *Secrets and variables* → *Actions*:

| Name | Kind | Value |
| --- | --- | --- |
| `HF_TOKEN` | secret | the Space write token |
| `HF_SPACE` | variable | `<hf-user>/<space>` |
| `VITE_API_BASE_URL` | variable | `https://<hf-user>-<space>.hf.space` |
| `VITE_SUPABASE_URL` | variable | `https://<ref>.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | variable | Supabase publishable key |

The last two are public by design (they end up in the browser); row-level
security and the backend's ownership checks are what protect the data.

*Settings* → *Pages* → *Source*: **GitHub Actions**.

## 6. Deploy

Push to `main`, or run both workflows from the *Actions* tab
(`workflow_dispatch`).

- **Deploy backend** runs the test suite, builds the image, and pushes it to
  the Space, which then builds it again (a few minutes).
- **Deploy frontend** lints, builds with the repository name as the base path,
  and publishes to `https://<user>.github.io/<repo>/`.

Check the API: `https://<hf-user>-<space>.hf.space/health?deep=true` should
report the database and storage as ok. `/api/llm/status` shows whether the
HPC model is reachable.

## 7. Day to day

- **HPC on:** `ssh iitd`, then `./run.sh --serve-llm`. It keeps the Mac awake,
  watches the tunnel, and resubmits the PBS job when its walltime ends.
  Ctrl-C turns the Funnel off; the job stays up, as with `./run.sh`.
- **HPC off:** do nothing. Chats use the cloud models until the HPC is back.
- **Update the app:** push to `main`. Only the changed half redeploys.
- **Local development:** `./run.sh` as before. Auth is off locally, so
  everything belongs to the single user `local`.

## 8. Checking it end to end

1. Open the site in a normal window: it loads without asking you to sign in.
   Upload a dataset, clean it, train a model, make a plot.
2. Reload: everything is still there.
3. Open the site in a private window: it starts empty, a separate user. The
   first window's dataset is not visible, and its ids return 404 from the API.
4. In the first window, *Save this project* → email. Open the link, then in
   a different browser *I have an account* → same email: the project appears.
5. Stop `run.sh --serve-llm`: the next chat still answers (cloud). Start it
   again: within about half a minute chats use the HPC model again.
6. *New project*: the dataset, models, plots and chat are gone, and the
   Supabase bucket has no files left under that user's id.

## Troubleshooting

- **Space build fine, but it cannot reach the database.** Hugging Face
  documents outbound traffic only on ports 80, 443 and 8080; the Supabase
  pooler has worked for others on 5432 but it is not guaranteed. Try the
  *Transaction pooler* URI (port 6543; the backend turns off prepared
  statements for it). If both are blocked, ask website@huggingface.co to allow
  the host, or move the API to another free Docker host.
- **First request after a quiet spell is slow.** Free Spaces sleep after 48 h
  without traffic and take a minute or two to wake.
- **"Your session has expired" (401).** The browser's token is older than the
  API accepts or Supabase keys were rotated; reloading signs in again. Check
  `SUPABASE_URL` matches the frontend's.
- **CORS errors in the browser console.** `CORS_ALLOW_ORIGINS` must be the
  exact origin, `https://<user>.github.io`, with no path or trailing slash.
- **Chats never use the HPC.** Open the Funnel URL plus `/models` in a browser:
  401 means the Funnel and vLLM are fine and the Space key is wrong; a
  Tailscale error page means `run.sh --serve-llm` is not running or Funnel is
  not enabled for the tailnet.
- **Cloud answers stop.** OpenRouter's free models allow about 50 requests a
  day per account (1000 with $10 of credit); add another provider's key.
- **Emails do not arrive.** The built-in Supabase sender is rate-limited; set
  up SMTP (step 2.3).

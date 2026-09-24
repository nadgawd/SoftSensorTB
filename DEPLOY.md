# Deploying online

The online app is three free pieces plus the HPC:

| Piece | Where | Holds |
| --- | --- | --- |
| Web app | GitHub Pages | The built React frontend, no secrets |
| API | Render (free Docker web service, `render.yaml`) | FastAPI backend, all server secrets |
| Data and sign-in | Supabase | Postgres, a private Storage bucket, Auth |
| Primary LLM | IITD HPC, through your Mac | vLLM behind an API key, reached via Tailscale Funnel |

Visitors get an anonymous account on first load, so their project is saved on
the server without any sign-up. It stays until they press **New project**,
which deletes their datasets, models and chat. "Save this project" in the
header attaches an email or Google login to the same account so they can come
back on another device.

When the HPC is not being served (your Mac asleep, the job between walltimes),
the API falls back to the cloud keys (OpenRouter, Gemini, Groq, Cerebras)
within a few seconds and keeps using them until the HPC answers again. The
website and API do not depend on your Mac at all.

Local development is unchanged: `./run.sh` needs none of this.

This project's values: GitHub `nadgawd/SoftSensorTB`, so the site is
`https://nadgawd.github.io/SoftSensorTB/`; Supabase project
`qqawziocwyecgzozbmll`.

---

## 1. GitHub repository

Already done: the code is on `main` at `github.com/nadgawd/SoftSensorTB`.
`.gitignore` keeps `.env`, datasets, the local database, `random/` and the
vLLM key out. GitHub Pages on a private repository needs GitHub Pro, which
you have.

## 2. Supabase

1. **Database URL.** *Connect* → *Session pooler* → copy the URI:
   `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`.
   URL-encode the password if it has symbols. This is `DATABASE_URL`. The
   backend switches it to asyncpg itself; do not add `sslmode`. (Already in
   your `.env`, second `DATABASE_URL` line, and tested.)
2. **Bucket and lockdown.** *SQL Editor* → paste `supabase/setup.sql` → *Run*.
   It creates the private `sst-data` bucket (already done). The backend creates
   its tables and enables row-level security on them at its first start; run
   the script once more afterwards and check every row reads
   `rls_enabled = true`, `policies = 0`.
3. **Auth.** *Authentication* →
   - *Sign In / Providers*: turn on **Allow anonymous sign-ins** and
     **Allow manual linking** (needed to attach Google to an anonymous user).
     Email stays on.
   - *URL Configuration*: Site URL `https://nadgawd.github.io/SoftSensorTB/`;
     add the redirect URLs `https://nadgawd.github.io/SoftSensorTB/**` and
     `http://localhost:5173/**`.
   - *Emails → SMTP*: the built-in sender allows only a few emails an hour.
     Add your own SMTP (e.g. Resend, Brevo) before sharing the app widely.
4. **Google (optional).** Create an OAuth client in Google Cloud (Web
   application, authorised redirect URI
   `https://qqawziocwyecgzozbmll.supabase.co/auth/v1/callback`) and paste its
   id and secret into the Google provider.
5. **Keys.** *Project Settings* → *API Keys*:
   - the **publishable** key (`sb_publishable_...`) goes to the frontend;
   - the **secret** key (`sb_secret_...`) goes only into Render, as
     `SUPABASE_SERVICE_ROLE_KEY`. It bypasses row-level security.

   The backend verifies user tokens against the project's JWKS, which works
   with the default asymmetric signing keys. Only if the project still signs
   with the legacy shared secret, also set `SUPABASE_JWT_SECRET` (*JWT Keys*
   page).

## 3. Tailscale Funnel (public URL for the HPC model)

Already set up on this Mac. The Funnel gives the Mac's forwarded vLLM port a
public HTTPS address; only the `/v1` path is published, and vLLM rejects any
request without the API key.

```bash
ssh iitd                    # authenticate once; leave it open
./run.sh --serve-llm
```

The model URL is `https://ninads-mac.tail328874.ts.net/v1`
(`LOCAL_LLM_BASE_URL`); the key is `LOCAL_LLM_API_KEY` in your `.env`, which
`run.sh` copies from the cluster.

On a new machine: install Tailscale (`brew install --cask tailscale`), sign in,
and approve Funnel with the link `run.sh` prints the first time.

## 4. Render (API)

`render.yaml` describes the service: free plan, Singapore region, Docker build
from `backend/Dockerfile`, health check `/health`, and deploys only commits
whose GitHub checks pass.

1. Sign up at [render.com](https://render.com) with GitHub.
2. *New* → *Blueprint* → give Render access to `nadgawd/SoftSensorTB` → pick
   the repo. Render reads `render.yaml` and asks for the secret values:

   | Name | Value |
   | --- | --- |
   | `DATABASE_URL` | Supabase session pooler URI |
   | `SUPABASE_SERVICE_ROLE_KEY` | Supabase secret key (`sb_secret_...`) |
   | `LOCAL_LLM_BASE_URL` | `https://ninads-mac.tail328874.ts.net/v1` |
   | `LOCAL_LLM_API_KEY` | `LOCAL_LLM_API_KEY` from your `.env` |
   | `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY` | from your `.env`; used when the HPC is off |

   Copy values without printing them, e.g.
   `grep '^LOCAL_LLM_API_KEY=' .env | cut -d= -f2- | tr -d '"' | pbcopy`.
   Render asks only on this first apply; change them later under the
   service's *Environment* tab.
3. *Apply*. The first build takes about five minutes. The API URL is shown on
   the service page, normally `https://soft-sensor-api.onrender.com` (Render
   adds a suffix if the name is taken).

`render.yaml` also sets `CORS_ALLOW_ORIGINS`, `SUPABASE_URL` and a 10 MB
upload limit. The image sets `AUTH_REQUIRED=true`, `STORAGE_BACKEND=supabase`
and `LLM_MODE=local_first`. Other optional limits (defaults in brackets):
`CHAT_RATE_LIMIT_PER_HOUR` (40 per user), `MAX_DATASETS_PER_USER` (3; uploading
a fourth deletes the oldest), `MAX_PROJECT_STATE_MB` (5).

**Free-plan limits.** 512 MB of RAM and a tenth of a CPU: fine for chat and
datasets of a few MB, slow for large trainings. The service sleeps after 15
minutes without requests; the next visitor waits about a minute (the page
says "Waking the server"). To keep it awake, add a free uptime monitor
(e.g. UptimeRobot) that requests `/health` every 10 minutes; one service
running all month fits in the 750 free hours.

## 5. GitHub settings

*Settings* → *Secrets and variables* → *Actions* → *Variables*:

| Name | Value |
| --- | --- |
| `VITE_API_BASE_URL` | the Render URL, e.g. `https://soft-sensor-api.onrender.com` |
| `VITE_SUPABASE_URL` | `https://qqawziocwyecgzozbmll.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | Supabase publishable key |

These are public by design (they end up in the browser); row-level security
and the backend's ownership checks are what protect the data.

*Settings* → *Pages* → *Source*: **GitHub Actions**.

## 6. Deploy

Push to `main`, or run the workflows from the *Actions* tab.

- **Backend CI** runs the test suite and builds and starts the Docker image.
  Render deploys the commit once it passes (backend changes only).
- **Deploy frontend** lints, builds with the repository name as the base path,
  and publishes to `https://nadgawd.github.io/SoftSensorTB/`.

Check the API: `<render url>/health?deep=true` should report the database and
storage as ok. `<render url>/api/llm/status` shows whether the HPC model is
reachable.

## 7. Day to day

- **HPC on:** `ssh iitd`, then `./run.sh --serve-llm`. It keeps the Mac awake,
  watches the tunnel, and resubmits the PBS job when its walltime ends.
  Ctrl-C turns the Funnel off; the job stays up, as with `./run.sh`.
- **HPC off / Mac asleep:** nothing to do. Chats use the cloud models until the
  HPC is back.
- **Update the app:** push to `main`. Only the changed half redeploys.
- **Local development:** `./run.sh` as before. Auth is off locally, so
  everything belongs to the single user `local`. Note that `.env` has two
  `DATABASE_URL` lines and the later (Supabase) one wins; comment one out.

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

- **Render deploy never starts.** It waits for the *Backend CI* checks on the
  commit; see the *Actions* tab. A commit that touches only the frontend does
  not redeploy the API.
- **Service restarts with "out of memory".** A large upload or training ran
  past 512 MB. Lower `MAX_UPLOAD_MB`, or move to Render's paid 2 GB instance.
- **"Your session has expired" (401).** The browser's token is older than the
  API accepts or Supabase keys were rotated; reloading signs in again. Check
  `SUPABASE_URL` matches the frontend's.
- **CORS errors in the browser console.** `CORS_ALLOW_ORIGINS` must be the
  exact origin, `https://nadgawd.github.io`, with no path or trailing slash.
- **Chats never use the HPC.** Open the Funnel URL plus `/models` in a browser:
  401 means the Funnel and vLLM are fine and the key in Render is wrong; a
  Tailscale error page means `run.sh --serve-llm` is not running.
- **Cloud answers stop.** OpenRouter's free models allow about 50 requests a
  day per account (1000 with $10 of credit); other providers' keys take over.
- **Emails do not arrive.** The built-in Supabase sender is rate-limited; set
  up SMTP (step 2.3).

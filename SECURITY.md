# Security Policy

## Supported versions

| Version | Status |
| ------- | ------ |
| `2.0.x` | patched for confirmed vulnerabilities |
| `1.3.x` | unsupported |

## Reporting a vulnerability

Please open a private security report on GitHub rather than a public issue.
Include the affected endpoint or script, a reproduction, and the impact you
believe it has. Expect acknowledgement within 5 business days and a status
update every 14 days until the report is resolved.

## Threat model for this repository

This is a research and demonstration service, not a production lending
system. It holds no customer data and exposes no authenticated identity
provider. The controls below limit the damage of the most likely mistakes.

### Trust boundaries

| Boundary | Assumption |
| --- | --- |
| `POST /predict`, `/explain` | Untrusted caller-derived input |
| `artifacts/model.joblib` | Untrusted file (pickle deserialization) |
| `data/raw/*.csv.gz` | Third-party download, verified by SHA-256 |
| `.env` | Secret-bearing, never committed |

### Controls in place

- **Model deserialization is path- and digest-checked.** `src/api/model.py`
  resolves any model path inside `LDR_ARTIFACTS_DIR` only and, when
  `artifacts/model_meta.json` records a `model_sha256`, refuses to load a file
  that does not match. This is the mitigation for pickle-based RCE, which is
  the highest-severity risk in any sklearn serving path.
- **Input is schema-constrained, not merely type-checked.** `LoanRequest`
  forbids unknown keys and bounds every field, for example `dti` in `[0, 50]`
  and `fico_range_low` in `[300, 850]`, so malformed payloads are rejected with
  422 before reaching the estimator.
- **The Docker build context excludes secrets and data.** `.dockerignore`
  keeps `.git`, `.env*`, `*.pem`, `*.key`, the 392MB raw archive, `.venv` and
  `artifacts/` out of the image, so a build cannot silently bake in a local
  credential or a stale model.
- **The API refuses to run without its declared dependencies.** `pydantic-settings`
  is a runtime dependency, not an optional one, so a clean install cannot import
  `src.loan_default_risk.config` and crash on container start.
- **No credential is ever written to the repository.** `.gitignore` covers
  `.env`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `token.json` and
  `credentials.json`. `.env.example` holds placeholders only.
- **Batch prediction is not exposed.** All endpoints are single-request, so a
  caller cannot amortize a large computation across a batch.

### Deliberately out of scope

- No authentication, authorisation or per-tenant isolation. The service is a
  public demo.
- No rate limiting or request throttling at the proxy layer. Deploy behind an
  ingress that provides one.
- No personally identifiable information is stored or logged.

## Secure deployment checklist

- [ ] `LDR_MODEL_PATH` points at a digest-verified artifact.
- [ ] `.env` is present on the host and absent from the image build context.
- [ ] Hugging Face / GitHub tokens live in secret storage, never in the remote
      URL or in a committed file.
- [ ] The service sits behind an HTTPS ingress with a rate limit.
- [ ] CI runs `bandit` and treats `B101` (assert used) and `B614` as failures.

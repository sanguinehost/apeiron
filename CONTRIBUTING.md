# Contributing to Sanguine Host

Sanguine Host is maintained by Lucas Rose-Winters ([@paperboygold](https://github.com/paperboygold)).
This policy applies to every repository in the [sanguinehost](https://github.com/sanguinehost)
organisation alongside a repository-specific section headed "This repository" with its build,
gates and paths. Where the two differ, the stricter rule applies. The canonical organisation policy
is <https://github.com/sanguinehost/.github/blob/main/CONTRIBUTING.md>.

## This repository

apeiron is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Report a vulnerability as
[SECURITY.md](SECURITY.md) says. The working agreement, with its gates and evidence grades, is
[AGENTS.md](AGENTS.md). It applies to people and coding agents alike. Read the
[book](book/src/introduction.md) chapter for the area you touch, and `proof/README.md` if you touch
the Lean. The deployment goal and its acceptance criteria are [GOAL.md](GOAL.md): a 7B-class model
at about 30 tok/s on the 6.6 GiB laptop with a 200k context, serving `../intent` and leaving 2 GiB
for the agent sessions. A pack, a row or a speedup is an artifact of that goal, not the goal.

### First run: build and check

Rust 1.89 or later is required (`rust-version` in `Cargo.toml`;
`cargo +1.89.0 check --lib --locked` passes). CI pins a nightly for lint stability and benchmark
comparability.

1. Open an issue first for anything larger than a small fix (see [Before you write code](#before-you-write-code)).
2. Build and run the contributor gate from the repository root:

```sh
cargo build --lib
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
python3 tools/check-claims.py
python3 tools/book-check.py
mdbook build book && mdbook test book -L target/debug/deps
```

3. If you change files under `proof/` or any Rust file included in the Lean extraction, also run the
   maintainer gate (see `proof/README.md` for the pinned toolchain):

```sh
proof/regen.sh --check
cd proof/corpus && lake build
```

4. Sign off every commit with `git commit -s` (see [Sign your work](#sign-your-work)), and add an
   `Assisted-by:` trailer if an AI tool materially shaped the commit (see [AI tools](#ai-tools)).

CI runs the same checks.

### A good change

- A result states its evidence grade (`proved`, `tested`, `measured`) and names the theorem, test or
  command behind it. A claim without one is not accepted.
- A fix comes with a test that fails before it and passes after.
- A speed claim comes with the command, the machine, and the mean over at least three runs with the
  spread.
- Documentation follows the same rules as code: it states what the source does, in the fewest words
  that keep every name and number, and it runs (`mdbook test book -L target/debug/deps`).
- Third-party source enters only as benchmark and comparison fixtures. They keep their own licence
  and stay outside the shipped library.

### Code owners

`.github/CODEOWNERS` gives each security surface its own entry: the GGUF reader (`src/format/`), the
engine (`src/engine/`), the GPU path (`src/gpu/`), the tables (`src/map/`), the cryptography
(`src/crypto/`), the proofs (`proof/`), CI (`.github/`) and the governance files.

## Before you write code

- Open an issue for anything larger than a small fix. Agree the problem and the approach first.
  A large pull request with no agreed issue may be closed unread.
- One concern per pull request, with the tests, documentation and receipt or benchmark it needs.
- Read the repository's `AGENTS.md`, `README.md` and the "This repository" section above. They are
  part of these terms.
- Do not include secrets, personal data, or code you do not have the right to submit.
- Do not copy code from a repository with an incompatible licence.

## Sign your work

Every commit in a contribution carries a `Signed-off-by:` line certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/):

```text
Signed-off-by: Your Name <you@example.com>
```

Use `git commit -s`. The sign-off states that you wrote the change, or have the right to submit it
under the repository's licence. Only a person signs off. CI refuses a pull request with an unsigned
commit.

## AI tools

Coding agents are welcome. The person submitting is responsible for the result.

- Only a human adds `Signed-off-by:`. An agent never does.
- Name the tools in an `Assisted-by:` trailer on each commit they materially shaped:
  `Assisted-by: <model> <tool>`.
- You must be able to explain and defend every line. A submission the author cannot explain is
  closed without detailed review.
- You check the provenance and licence of generated code as if you had written it.
- An AI system never approves, merges or closes a contribution, and is never the final word on one.

The maintainer's own work uses coding agents under his direction and his gates. Those commits are
made in his name, and he answers for them.

## Response times

Times are working days, Monday to Friday, Australia/Hobart.

- An automated triage pass labels and ranks new issues and pull requests. Its comments say they are
  automated. It does not count as a response.
- A human replies to a new issue or pull request within one working day. This is a target, not a
  guarantee, and the queue's first-response time is published quarterly.
- A pull request with no activity from its author for 90 days may be closed.

## Security

Report a suspected vulnerability privately through the repository's GitHub private vulnerability
reporting (Security, then Report a vulnerability), or as its `SECURITY.md` says. Never report one
in a public issue or pull request. Reports are acknowledged within 3 days and fixed or published
within 90 days. The project signs no non-disclosure agreement and treats a vulnerability found
with AI tools as already public. Releases are built from git tags signed on the top maintainer's
hardware key. No release artefact contains code that is not in the tagged tree.

## Conduct

Review is technical and direct. Disagreement about the code is welcome. Harassment, threats,
doxxing and sustained disruption are not. Closing or rejecting a contribution is not a conduct
violation. The top maintainer may close a contribution, remove a role, or block an account. When
there is more than one maintainer, a complaint about the top maintainer goes to the subsystem
maintainers, who answer it together.

## Open in perpetuity

- Every release stays under the licence it shipped with. An open-source licence grant cannot be
  withdrawn, so a released version stays open whatever happens to the project later.
- Contributions are accepted under the repository's existing `LICENSE` (inbound = outbound). You keep
  copyright in what you write. There is no contributor licence agreement and no copyright
  assignment, and the project will not adopt either. Once code from other people lands, the licence
  can only change with every copyright holder's consent.
- The project names and logos are not licensed by the code licence (Apache-2.0 section 6). A fork is
  welcome under the licence. It may not present itself as the official project.
- A repository with no `LICENSE` accepts no code. Open an issue and ask for the terms first.

## Maintainers

The project uses the Linux model: one top maintainer integrates, and subsystem maintainers own
paths below him.

| Role | Holds | Granted when |
|---|---|---|
| Top maintainer | `main`, releases, signing keys, organisation admin, this policy | @paperboygold |
| Subsystem maintainer | merge for named paths in `CODEOWNERS`; not `main` outside them, not releases | two approvers in one area and required CI green on the path |
| Approver | approval for named paths; the top maintainer still merges | 10 merged pull requests, 3 months as a reviewer, and CI covering the path |
| Reviewer | review comments and `Reviewed-by:` trailers; no write access | 3 merged pull requests and 2 substantive reviews |
| Triager | labels, assignment, duplicates; no write, approval or merge | the first recurring outside contributor, or a queue the top maintainer cannot clear for two consecutive months |

Today the top maintainer holds every role. The thresholds are the minimum. Meeting them makes a
person eligible. It does not entitle them to the role.

Rights are personal:

- A role is granted by the top maintainer, in writing, as a change to this file and `CODEOWNERS`.
  Contributing, sponsoring, friendship, employment or discussion grant nothing.
- A role belongs to one person and one account. It cannot be shared, lent, transferred or
  sub-delegated. Merging, approving or pushing on someone else's behalf is grounds for removal.
- An account holding any role, the top maintainer's included, uses a hardware security key (FIDO2,
  such as a YubiKey) as its GitHub second factor, with no SMS or authenticator-app fallback. Its
  commits and tags are signed with a key held on that hardware key (an `ed25519-sk` SSH key or an
  OpenPGP key on the device).
- A role is removed after 6 months without activity, with 4 weeks' notice. It is removed at once,
  without notice, on a security concern or a breach of this section.
- A pull request is a proposal. It transfers no ownership, decision right or access. Any
  maintainer may accept, revise, defer or close one in their paths without giving a reason.

## Enforcement

- Organisation membership is limited to the top maintainer until this file grants a role. There are
  no outside collaborators and no teams with repository access.
- Every repository carries `.github/CODEOWNERS`. Each security surface has its own entry.
- Public repositories require a pull request, code-owner review, signed-off commits and green CI on
  the default branch, block force pushes and deletion, and give the top maintainer the only bypass.
  Private repositories on the GitHub Free plan have no branch rules; there only the top maintainer
  has write access.
- Installed apps and agents hold the narrowest permission their job needs. The triage agent reads
  issues and writes labels and comments. It holds no contents, pull-request merge or workflow
  permission, because issue text is untrusted input.
- The top maintainer audits members, collaborators, teams, deploy keys and app installations before
  any access change and after it.
- Forks of upstream projects are not maintained here. Send changes upstream.

## Succession

The organisation, the names, the domain and the signing keys are held by Lucas Rose-Winters
personally. A sealed note lists them. If he dies or loses capacity, his executor or enduring
attorney transfers them to a non-profit open-source foundation named in that note, never to an
individual. The foundation admits members and contributors from every country. The licences
above do not change.

Only the top maintainer changes this policy.

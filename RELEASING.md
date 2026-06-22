# Releasing transcribe.cpp

The version is authored once in `include/transcribe.h`
(`TRANSCRIBE_VERSION_{MAJOR,MINOR,PATCH}`) and duplicated across ~14 files.
`scripts/release/prepare.py` writes all of them and regenerates the FFI;
`prepare.py --check` verifies the whole tree is consistent. A git tag matching
`v[0-9]*` is the **only** trigger for the release pipeline (`publish.yml`), and
the tag run does not re-run the drift gates — so **only ever tag a commit that
has gone green through branch CI on `main`.**

## Cutting a release (e.g. `0.0.x`)

```bash
# 1. Branch and write the bump everywhere (header, Cargo/npm/py manifests,
#    both lockfiles, Swift, and the regenerated FFI) in one command.
git switch -c release-0.0.x
uv run --no-project scripts/release/prepare.py 0.0.x

# 2. Verify the tree is release-consistent. Only the version-bump files should
#    show as modified; --check must not introduce anything else.
uv run --no-project scripts/release/prepare.py --check
git status --porcelain

# 3. Commit the bump and push the branch.
git commit -am "release: 0.0.x"
git push -u origin release-0.0.x

# 4. Open a PR. Wait for ALL branch CI green
#    (rust-ci, python-bindings, typescript-ci, swift-ci, native-ci), then merge.
#    Branch CI is the real gate — the tag run does NOT re-check it.

# 5. (optional) Dry run before tagging: full build + TestPyPI, no prod registries.
gh workflow run publish.yml -f version=0.0.x

# 6. After the PR merges, tag the merged commit on main and push the tag.
#    Pulling main puts HEAD on the merge commit, so you tag exactly what CI
#    proved. The tag string MUST equal the prepared version. The push publishes.
git checkout main && git fetch origin && git pull
git tag -a v0.0.x -m "transcribe.cpp v0.0.x"
git push origin v0.0.x
```

# Development

## Commands

```bash
cargo develop && cargo test && pytest -q
ship-rs-build
```

`cargo develop` builds the Python extension with the `test` profile and puts it in `python/aihist/`, where the editable install finds it.

## Layout

The root package is a pure Rust crate, published to crates.io. Rust users who depend on it never build PyO3. Its modules are:

- `cells`: one notebook cell as a dialog message, and as a model reads it
- `input`: a prompt's model input, with the examples, CRAFTs, `$` values and warnings that a host supplies
- `output`: code outputs and tool results as a model reads them
- `media`: MIME detection, resizing images, and media tags
- `tokens`: approximate token counts
- `reply`: reading and writing saved replies

The Python extension is the `py` package. It depends on the root crate, is never published, and maturin builds it through `[tool.maturin].manifest-path`. Its functions take and return JSON as strings. The `aihist` Python package in `python/` converts aidialog objects to that JSON and back, in `aihist.reply` and `aihist.hist`. `tests/*.py` tests the Python API, and `cargo test` runs the crate's unit tests.

aihist keeps aidialog's notebook and saved-reply formats. aidialog writes saved replies for fastllm, so a saved reply has two writers, aidialog's `hist2fmt` and this crate's `reply::write`. `tests/test_basic.py` checks that they write the same blocks.

## Versioning

The canonical version lives in `[workspace.package]` in `Cargo.toml`, which both crates inherit. `pyproject.toml` gets the Python package version from Cargo via `dynamic = ["version"]`.

## Build profiles

uv builds and `maturin develop --release` use the incremental `release` profile for fast local iteration. `cargo test` and `cargo develop` build the `test` profile, which inherits `release`. CI builds distributed wheels with `dist`, which enables full LTO and one codegen unit, disables incremental compilation, and strips the result.

## Release

1. Run `cargo develop && cargo test && pytest -q`.
2. Confirm the release version in `Cargo.toml` (`[workspace.package].version`).
3. Run `ship-release`.

`ship-release` pushes a `v<version>` tag, which starts the release workflow in GitHub Actions, then bumps the version. On a version tag, the workflow builds the wheels and sdist, then publishes the crate to crates.io and the wheels and sdist to GitHub and PyPI. For the first release, publish the crate manually with a token (`cargo publish`), then add this repo's `ci.yml` as a trusted publisher in the crate's crates.io settings. Later tags publish through CI without a token.

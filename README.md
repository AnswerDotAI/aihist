# aihist

PyO3/maturin package scaffolded by fastship. The Rust crate is in `src/`, and its Python extension in `py/`.

## Development

```bash
pip install -e .[dev]
maturin develop && pytest -q
```

## Build

```bash
ship-rs-build
```

## Release

```bash
maturin develop && pytest -q
ship-release
```

`ship-release` pushes a `v<version>` tag, which starts the release workflow in GitHub Actions, then bumps the version. On a version tag, the workflow builds the wheels and sdist, then publishes the crate to crates.io and the wheels and sdist to GitHub and PyPI. For the first release, publish manually with a token (`cargo publish`), then add this repo's `ci.yml` as a trusted publisher in the crate's crates.io settings. Later tags publish through CI without a token.

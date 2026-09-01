# Aither World Router (awrouter)

The stateless LLM routing plane — tier/capability/thinking routing with
health-probed failover, extracted from the AitherOS MicroScheduler core.

Source mirror of the `awrouter` package. The publish lane: this repo's
`pypi-publish` workflow (trusted publishing to PyPI). The sync lane lives
in the monorepo (`sync-awrouter.yml`), which boundary-scans the source
before it leaves (zero monorepo imports, zero internal identifiers).

```bash
pip install awrouter
```

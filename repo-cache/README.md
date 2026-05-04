# Optional Upstream Repository Cache

Runtime no longer depends on this directory. Preview algorithms are bundled under
`backend/app/preview/vendor`, and worker images are built from project code.

Use this cache only when maintainers need to compare or refresh vendored source
against the fixed upstream commits. Large cloned repositories are intentionally
ignored by Git.

# News fragments

This directory holds **unreleased** changelog entries, one file per change.
At release time [towncrier](https://towncrier.readthedocs.io/) collects them into
`CHANGELOG.md` and deletes them.

## Add an entry

Every user-facing PR should add a fragment. Filename format:

```
<issue-or-pr>.<type>.md
```

`<type>` is one of:

| Type         | Section in changelog | Use for                                  |
| ------------ | -------------------- | ---------------------------------------- |
| `added`      | Added                | New features, datasets, public API       |
| `changed`    | Changed              | Changes to existing behaviour/API        |
| `deprecated` | Deprecated           | Soon-to-be-removed features              |
| `removed`    | Removed              | Removed features/API                     |
| `fixed`      | Fixed                | Bug fixes                                |
| `security`   | Security             | Security-relevant fixes                  |

The body is a single Markdown bullet describing the change for **end users**.

### Examples

```bash
# tied to PR/issue #42
towncrier create -c "Add hourly variant of the ETT dataset." 42.added.md

# no issue number — use the orphan prefix '+'
towncrier create -c "Fix NPZ cache invalidation when the scaler changes." +cache-fix.fixed.md
```

Or just create the file by hand:

```
changelog.d/42.added.md   ->   "Add hourly variant of the ETT dataset."
```

## Preview the next release

```bash
towncrier build --draft --version <next-version>
```

> This `README.md` is ignored by towncrier (its extension is not a fragment type).

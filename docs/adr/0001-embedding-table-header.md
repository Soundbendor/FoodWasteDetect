# Embedding tables carry a store-owned header; legacy or incompatible tables are refused, never migrated

Status: accepted (2026-11-06)

Embedding tables take hours or paid Vertex requests to build, and tables written
before this decision recorded only an optional creation snapshot, so a table built
under one configuration could be resumed, evaluated, or derived from under
another without any error. We decided that every embedding table records a header
owned by the embedding table module when it is created: a schema version, the
role (reference table or validation cache), the dimension, and, for a prefix
table, the source table and source dimension. The header is never rewritten. The
experiment's creation snapshot is stored beside it verbatim and is never parsed
or rewritten, including for prefix tables.

A table without a valid header, or with another schema version, is refused with
an error telling the researcher to rebuild it under a new name. It is not read,
repaired, or migrated. Later integrity checks that find a table incompatible with
the current configuration follow the same rule.

## Considered options

- **Migrate legacy tables in place.** Rejected: old tables carry no reliable record
  of their encoder or inputs, so a migration would have to guess, and a wrong guess
  silently produces meaningless scores. Rewriting a table that cost paid requests
  also risks the data it was meant to preserve.
- **Read legacy tables without a header.** Rejected: every caller would need a
  second code path, and the failures the header exists to catch would stay silent
  for those tables.
- **An override flag to accept a mismatched table.** Rejected: each table should
  describe exactly one configuration; choosing a new table name is cheap and
  explicit.

## Consequences

Every table written before this decision, including the paid Gemini Embedding 2
reference tables and validation caches, must be rebuilt under a new name before it
can be used. Bumping the schema version later makes every table written under the
earlier version legacy in the same way, so a version bump is a deliberate,
budgeted rebuild.

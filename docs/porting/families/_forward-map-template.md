# Forward map - <family>

Reference: <source path or package> @ <revision>
Closest in-tree analog: src/arch/<family>/

A compact map from the reference forward pass to the C++ port. Family-level
because most variants share architecture; variant differences go in the
"Variant Notes" section only when they affect graph shape, control flow,
capabilities, or validation coverage.

For repeated layers, map the block pattern once and list the first / middle /
last gate tensors. One row per layer is unnecessary unless the architecture
actually changes across layers.

This file is a Stage 4 planning artifact. It is created (or updated for
a sibling variant) before C++ implementation work begins, but `TODO` /
`UNKNOWN` / `accepted_gap` entries are fine during bring-up — they are
how the map earns its keep. The "no unresolved rows" gate applies only
at Stage 4 sign-off, unless the user explicitly accepted the gap in
intake or the family doc.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|

## Decoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|

## Deviations From Closest Analog

- <op or shape that differs from the in-tree analog and how the port handles it>

## Variant Notes

- `<variant>`: <"same architecture as family baseline" | concrete delta>

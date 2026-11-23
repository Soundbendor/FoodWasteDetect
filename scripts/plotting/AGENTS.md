# Dissertation figures

Use [style.py](style.py) for the print layout, type sizes, grid and export
settings. Apply the style through the save call. For an accuracy plot:

```python
with style.dissertation_style():
    fig, ax = style.accuracy_figure()
    ax.plot(x, accuracy_pct, **style.model_style(model_id))
    ax.set(xlabel=x_label, title=title)
    style.add_model_legend(fig)
    fig.savefig(output)
```

- Register each named model in `MODEL_STYLES` with a display label, color and
  distinct marker. The marker must be legible in grayscale. Unknown IDs use
  their raw labels and deterministic fallback styles.
- Use short sentence-case titles. Put experiment details in the LaTeX caption.
  Mark partial results in the title and break lines at missing scheduled runs.
- Plot accuracy in percentage points and label ticks with percent signs. Use
  the full range for image counts and a padded observed range for dimensions.
  Preserve numeric spacing for dimensions and ordered cap categories for counts.
- Use vector PDF for LaTeX and PNG for raster output. Check both sweeps at
  final print size for overlapping ticks, clipped legends and grayscale legibility.
  Run the [plotting tests](../../docs/development.md#report-plotting) with Pixi.

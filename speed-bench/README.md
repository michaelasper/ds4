# LagoonNebula speed benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `lgn2-bench` as:

```
./lgn2-bench \
  -m lgn2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/lgn2-speed.csv
```

Provide PR including your numbers if your hardware was not already tested.
The command writes the generated benchmark CSV to `/tmp/lgn2-speed.csv`.
Preserve a copy with a hardware-specific name when submitting results.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py /tmp/lgn2-speed.csv --title "Apple Metal t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `/tmp/lgn2-speed_ts.svg`.

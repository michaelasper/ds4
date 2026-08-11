# LagoonNebula Laguna S 2.1 model and runtime note

This checkout uses one pinned Laguna S 2.1 GGUF for its benchmark and local
Metal runtime. The required model identity is:

```text
File:     laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
SHA-256:  61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903
```

The model is approximately 45 GiB, so the benchmark runbook asks for at least
55 GiB free on the model volume. The downloader fetches the file from the
locally recorded Hugging Face source
`https://huggingface.co/antirez/Laguna-S-2.1-GGUF/resolve/main/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf`
and verifies the pinned SHA-256 before installing it.

```sh
export LGN2_GGUF_DIR=/absolute/path/to/laguna-models
./download_model.sh laguna-q2-q3
```

The downloader resumes its own `.part` file, leaves existing model files and
the `lgn2.gguf` convenience link untouched, and refuses a model path whose hash
does not match the benchmark identity. Use the verified file at
`$LGN2_GGUF_DIR/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` directly when a
convenience link must remain unchanged. No legacy model-link name is consulted.

The current product boundary is Laguna S 2.1 on Apple Metal. The benchmark
runner separately creates its own read-only input symlink and rechecks the
model hash and file identity before and after every timed process.

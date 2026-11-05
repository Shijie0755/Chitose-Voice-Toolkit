# Chitose v0.1.0 model card

## Release files

| File | Size | SHA-256 |
|---|---:|---|
| <code>Chitose-v0.1.0-SoVITS.pth</code> | 85,008,426 bytes | <code>529D0040F4DC1CD54854863F6132B14FB09918378853CB9C1EC10FF517DB0C8F</code> |
| <code>Chitose-v0.1.0-GPT.ckpt</code> | 155,313,504 bytes | <code>CC011B555DB21082B3C8F06192E5868EC6A5B63B22603C8CA1E7EEAFB58536D7</code> |

Both files are required by the GPT-SoVITS v2 inference configuration.

## Training snapshot

- Dataset scope: EP01–EP02 retained clips
- Clips: 154
- Duration: 580.332 seconds / 9.672 minutes
- Audio: mono, 32000 Hz, 16-bit PCM
- Denoise filter: <code>afftdn=nr=6:nf=-48:tn=1:gs=5</code>
- SoVITS training: 8 epochs
- GPT training: 15 epochs

The training audio and subtitle-derived transcripts are not distributed in this repository.

## Limitations

This is an early personal research checkpoint. It may produce unstable pronunciation, prosody artifacts, background characteristics inherited from reference audio, and results that vary strongly with prompt audio. It is not an official model and must not be used to impersonate the original voice actor or mislead listeners.


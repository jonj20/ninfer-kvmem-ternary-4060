# Windows binary installer

PowerShell scripts that install and update the NInfer Windows x64 release binaries published at
<https://github.com/iamwavecut/ninfer-all/releases>. They run in Windows PowerShell 5.1 (stock
Windows 10/11) and PowerShell 7.

## Requirements

- Windows x64
- [7-Zip](https://www.7-zip.org/) (`winget install 7zip.7zip`); `7z.exe` on `PATH` or in its default
  install folder
- `curl.exe` (ships with Windows 10 1803 and newer)
- an NVIDIA driver and the Microsoft Visual C++ x64 runtime, as described in
  [release-archive-windows.md](../../docs/release-archive-windows.md)

## install.ps1

```powershell
.\install.ps1                    # first run: pick a flavor and install; later runs: update if newer
.\install.ps1 -CheckOnly         # report only; exit code 2 when an update is available
.\install.ps1 -Flavor cuda13     # switch flavor (saved for later updates)
.\install.ps1 -Version 2026-10-28.1 -Force   # install or reinstall a specific release
.\install.ps1 -Yes               # no prompts (unattended)
```

If script execution is blocked, run it with
`powershell -ExecutionPolicy Bypass -File .\install.ps1`.

| Flavor | Package | Use |
|---|---|---|
| `dual` | Dual CUDA 12.9 + 13.4 (largest) | default; picks the CUDA backend for the GPU and driver |
| `cuda12` | CUDA 12.9 | Volta/V100, or a driver without CUDA 13 support |
| `cuda13` | CUDA 13.4 | Blackwell/RTX 50, or a CUDA 13 driver on Turing to Ada |

Parameters: `-InstallDir` (default: this script's folder), `-Flavor dual|cuda12|cuda13`,
`-Version <tag>`, `-Yes`, `-CheckOnly`, `-Force`, `-Repo <owner/name>`. Set `GITHUB_TOKEN` to avoid
the anonymous GitHub API rate limit.

What it does:

1. checks for Windows x64, 7-Zip and `curl.exe`;
2. reads the installed version and flavor from `ninfer-install.json`, or asks for a flavor;
3. compares the installed release tag with the latest release (or `-Version`) and asks before
   installing;
4. downloads every `.7z.NNN` part of the flavor's package into `.staging\downloads` (resumable: an
   interrupted run continues where it stopped) and verifies each part's size and SHA-256 against the
   release's `SHA256SUMS-*.txt` (or GitHub's asset digest);
5. extracts the first part with 7-Zip into `.staging\extract` and verifies the files against the
   package's internal `SHA256SUMS.txt`;
6. renames the current `bin` aside and moves the new package into `bin` (restoring the old one if
   the move fails), records the new state, then deletes the old `bin` and `.staging`.

Stop running `ninfer-serve.exe` and other NInfer processes before updating; a locked `bin` folder
aborts the swap and keeps the installed version.

## Layout

```
install.ps1
ninfer-install.json   installed repo, flavor, version (release tag), commit, package, install time
bin\                  release contents: launchers, backends\, help\, BUILD-INFO.json, ...
.staging\             present only during an install
```

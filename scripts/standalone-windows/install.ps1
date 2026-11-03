<#
.SYNOPSIS
    Installs or updates the NInfer Windows x64 binary release into <InstallDir>\bin.

.DESCRIPTION
    Checks the GitHub releases of the NInfer repository, compares the newest release with the
    installed one recorded in ninfer-install.json, and after confirmation downloads every 7-Zip
    volume of the selected package flavor, verifies their SHA-256 digests, extracts them with 7-Zip
    and replaces the bin folder in one rename. The selected flavor is remembered for later updates.

.PARAMETER InstallDir
    Install root. Defaults to the folder that contains this script.

.PARAMETER Flavor
    dual (CUDA 12.9 + 13.4), cuda12 (CUDA 12.9) or cuda13 (CUDA 13.4). Changes the saved flavor.

.PARAMETER Version
    Install this release tag instead of the latest release.

.PARAMETER Yes
    Do not ask for confirmation. Without a saved flavor or -Flavor, selects dual.

.PARAMETER CheckOnly
    Report whether an update is available and exit (exit code 2 when one is).

.PARAMETER Force
    Reinstall even when the installed version and flavor already match.

.PARAMETER Repo
    GitHub repository that publishes the releases.

.EXAMPLE
    .\install.ps1
.EXAMPLE
    .\install.ps1 -Flavor cuda13 -Yes
.EXAMPLE
    .\install.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [string]$InstallDir,
    [ValidateSet('dual', 'cuda12', 'cuda13')]
    [string]$Flavor,
    [string]$Version,
    [switch]$Yes,
    [switch]$CheckOnly,
    [switch]$Force,
    [string]$Repo = 'iamwavecut/ninfer-all'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$StateFileName = 'ninfer-install.json'

$Flavors = [ordered]@{
    dual   = @{
        Label   = 'Dual CUDA 12.9 + 13.4'
        Hint    = 'Default. Both backends; picks CUDA 12 or 13 for the GPU and driver.'
        Pattern = '^ninfer-windows-x64-cuda12\.\d+-13\.\d+-[0-9a-f]{7,40}\.7z\.(?<part>\d{3})$'
    }
    cuda12 = @{
        Label   = 'CUDA 12.9'
        Hint    = 'Smaller. Volta/V100, or a driver that supports CUDA 12 but not CUDA 13.'
        Pattern = '^ninfer-windows-x64-cuda12\.\d+-sm\d+a?(?:-\d+a?)*-[0-9a-f]{7,40}\.7z\.(?<part>\d{3})$'
    }
    cuda13 = @{
        Label   = 'CUDA 13.4'
        Hint    = 'Smaller. Blackwell/RTX 50, or a CUDA 13 driver on Turing to Ada.'
        Pattern = '^ninfer-windows-x64-cuda13\.\d+-sm\d+a?(?:-\d+a?)*-[0-9a-f]{7,40}\.7z\.(?<part>\d{3})$'
    }
}

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Note([string]$Message) { Write-Host "    $Message" }
function Write-Warn([string]$Message) { Write-Host "WARNING: $Message" -ForegroundColor Yellow }

function Format-Size([double]$Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N2} GiB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N1} MiB' -f ($Bytes / 1MB)) }
    return ('{0:N0} bytes' -f $Bytes)
}

function Confirm-Action([string]$Question) {
    if ($Yes) { return $true }
    $answer = Read-Host "$Question [y/N]"
    return ($answer -match '^(y|yes)$')
}

function Assert-Platform {
    if ($env:OS -ne 'Windows_NT') { throw 'This installer supports Windows only.' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This installer supports Windows x64 only.' }
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        throw 'This installer supports Windows x64 only (ARM64 detected).'
    }
}

function Find-SevenZip {
    foreach ($name in @('7z.exe', '7za.exe')) {
        $cmd = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmd) { return $cmd.Source }
    }
    $candidates = @()
    if ($env:ProgramFiles) { $candidates += (Join-Path $env:ProgramFiles '7-Zip\7z.exe') }
    if (${env:ProgramFiles(x86)}) { $candidates += (Join-Path ${env:ProgramFiles(x86)} '7-Zip\7z.exe') }
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\7-Zip\7z.exe') }
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { return $path }
    }
    throw ("7-Zip was not found. Install it (for example: winget install 7zip.7zip) " +
        "or put 7z.exe on PATH, then run this script again.")
}

function Find-Curl {
    $system = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (Test-Path -LiteralPath $system -PathType Leaf) { return $system }
    $cmd = Get-Command 'curl.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    throw 'curl.exe was not found (it ships with Windows 10 1803 and newer).'
}

function Read-State([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try {
        return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    } catch {
        Write-Warn "Ignoring unreadable $Path : $($_.Exception.Message)"
        return $null
    }
}

function Get-StateValue($State, [string]$Name) {
    if ($null -eq $State) { return $null }
    $prop = $State.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

function Write-State([string]$Path, [hashtable]$Data) {
    $tmp = "$Path.tmp"
    $json = ([pscustomobject]$Data) | ConvertTo-Json -Depth 4
    [IO.File]::WriteAllText($tmp, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Select-Flavor([string]$Current) {
    $keys = @($Flavors.Keys)
    Write-Host ''
    Write-Host 'Select the package flavor:'
    for ($i = 0; $i -lt $keys.Count; $i++) {
        $f = $Flavors[$keys[$i]]
        $mark = ''
        if ($keys[$i] -eq $Current) { $mark = ' (current)' }
        Write-Host ("  {0}) {1}{2}" -f ($i + 1), $f.Label, $mark)
        Write-Host ("     {0}" -f $f.Hint)
    }
    $default = 1
    if ($Current) { $default = [array]::IndexOf($keys, $Current) + 1 }
    while ($true) {
        $answer = Read-Host "Choice [$default]"
        if ([string]::IsNullOrWhiteSpace($answer)) { return $keys[$default - 1] }
        $n = 0
        if ([int]::TryParse($answer.Trim(), [ref]$n) -and $n -ge 1 -and $n -le $keys.Count) {
            return $keys[$n - 1]
        }
        Write-Host 'Enter one of the listed numbers.'
    }
}

function Invoke-GitHubApi([string]$Path) {
    $headers = @{ 'User-Agent' = 'ninfer-install-ps1'; 'Accept' = 'application/vnd.github+json' }
    if ($env:GITHUB_TOKEN) { $headers['Authorization'] = "Bearer $($env:GITHUB_TOKEN)" }
    $uri = "https://api.github.com/repos/$Repo/$Path"
    try {
        return Invoke-RestMethod -Uri $uri -Headers $headers -UseBasicParsing
    } catch {
        throw "GitHub API request failed ($uri): $($_.Exception.Message)"
    }
}

function Get-Release {
    if ($Version) { return Invoke-GitHubApi "releases/tags/$Version" }
    return Invoke-GitHubApi 'releases/latest'
}

function ConvertTo-TagKey([string]$Tag) {
    if ($Tag -match '^v?(\d{4})-(\d{2})-(\d{2})\.(\d+)$') {
        return [long]($Matches[1] + $Matches[2] + $Matches[3]) * 100000 + [long]$Matches[4]
    }
    return $null
}

# 1 when $Candidate is newer, 0 when equal, -1 when older, $null when the tags are not comparable.
function Compare-Tag([string]$Installed, [string]$Candidate) {
    if ($Installed -eq $Candidate) { return 0 }
    $a = ConvertTo-TagKey $Installed
    $b = ConvertTo-TagKey $Candidate
    if ($null -eq $a -or $null -eq $b) { return $null }
    if ($b -gt $a) { return 1 }
    if ($b -lt $a) { return -1 }
    return 0
}

function Resolve-Package($Release, [string]$FlavorKey) {
    $pattern = $Flavors[$FlavorKey].Pattern
    $parts = @($Release.assets | Where-Object { $_.name -match $pattern })
    if ($parts.Count -eq 0) {
        throw "Release $($Release.tag_name) has no '$($Flavors[$FlavorKey].Label)' Windows x64 package."
    }
    $bases = @($parts | ForEach-Object { $_.name -replace '\.7z\.\d{3}$', '' } | Sort-Object -Unique)
    if ($bases.Count -ne 1) {
        throw "Release $($Release.tag_name) has several matching packages: $($bases -join ', ')"
    }
    $base = $bases[0]
    $parts = @($parts | Sort-Object { [int](($_.name -split '\.')[-1]) })
    for ($i = 0; $i -lt $parts.Count; $i++) {
        $expected = '{0}.7z.{1:D3}' -f $base, ($i + 1)
        if ($parts[$i].name -ne $expected) {
            throw "Release $($Release.tag_name) is missing archive part $expected."
        }
    }
    $manifest = $Release.assets | Where-Object { $_.name -eq "SHA256SUMS-$base.txt" } | Select-Object -First 1
    $total = 0L
    foreach ($p in $parts) { $total += [long]$p.size }
    $commit = $null
    if ($base -match '-([0-9a-f]{7,40})$') { $commit = $Matches[1] }
    return [pscustomobject]@{
        Base     = $base
        Parts    = $parts
        Manifest = $manifest
        Total    = $total
        Commit   = $commit
    }
}

# Windows PowerShell 5.1 turns redirected native stderr into terminating errors under 'Stop';
# curl and 7-Zip write progress there, so success is judged by exit code only.
function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments | Out-Host
    } finally {
        $ErrorActionPreference = $saved
    }
    return $LASTEXITCODE
}

function Invoke-Download([string]$Curl, [string]$Url, [string]$Path) {
    $code = Invoke-Native $Curl @('-L', '--fail', '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30',
        '--progress-bar', '-C', '-', '-o', $Path, $Url)
    if ($code -ne 0) { throw "Download failed (curl exit code $code): $Url" }
}

function Read-Sha256Manifest([string]$Path) {
    $map = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$') {
            $name = $Matches[2] -replace '^\./', ''
            $map[$name] = $Matches[1].ToLowerInvariant()
        }
    }
    return $map
}

function Get-AssetDigest($Asset) {
    $prop = $Asset.PSObject.Properties['digest']
    if ($null -ne $prop -and $prop.Value -match '^sha256:([0-9a-fA-F]{64})$') {
        return $Matches[1].ToLowerInvariant()
    }
    return $null
}

function Get-PackageParts($Package, [string]$DownloadDir, [string]$Curl) {
    New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

    $keep = @($Package.Parts | ForEach-Object { $_.name })
    if ($Package.Manifest) { $keep += $Package.Manifest.name }
    Get-ChildItem -LiteralPath $DownloadDir -File | Where-Object { $keep -notcontains $_.Name } |
        Remove-Item -Force

    $digests = @{}
    if ($Package.Manifest) {
        $manifestPath = Join-Path $DownloadDir $Package.Manifest.name
        Remove-Item -LiteralPath $manifestPath -Force -ErrorAction SilentlyContinue
        Invoke-Download $Curl $Package.Manifest.browser_download_url $manifestPath
        $digests = Read-Sha256Manifest $manifestPath
    } else {
        Write-Warn 'The release has no SHA256SUMS manifest for this package; using GitHub asset digests.'
    }

    $index = 0
    foreach ($part in $Package.Parts) {
        $index++
        $path = Join-Path $DownloadDir $part.name
        $size = [long]$part.size
        $label = "[$index/$($Package.Parts.Count)] $($part.name) ($(Format-Size $size))"
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $have = (Get-Item -LiteralPath $path).Length
            if ($have -gt $size) { Remove-Item -LiteralPath $path -Force }
        }
        if ((Test-Path -LiteralPath $path -PathType Leaf) -and (Get-Item -LiteralPath $path).Length -eq $size) {
            Write-Note "$label already downloaded"
        } else {
            Write-Note "Downloading $label"
            Invoke-Download $Curl $part.browser_download_url $path
        }
        $have = (Get-Item -LiteralPath $path).Length
        if ($have -ne $size) {
            Remove-Item -LiteralPath $path -Force
            throw "$($part.name) is $have bytes; the release lists $size bytes."
        }

        $expected = $null
        if ($digests.ContainsKey($part.name)) { $expected = $digests[$part.name] }
        if (-not $expected) { $expected = Get-AssetDigest $part }
        if ($expected) {
            Write-Note "Verifying SHA-256 of $($part.name)"
            $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) {
                Remove-Item -LiteralPath $path -Force
                throw "SHA-256 mismatch for $($part.name) (expected $expected, got $actual). The part was deleted; run again to download it anew."
            }
        } else {
            Write-Warn "No SHA-256 digest is published for $($part.name); size check only."
        }
    }
    return (Join-Path $DownloadDir $Package.Parts[0].name)
}

function Get-PackageRoot([string]$ExtractDir) {
    $root = $ExtractDir
    while ($true) {
        $files = @(Get-ChildItem -LiteralPath $root -File -Force)
        $dirs = @(Get-ChildItem -LiteralPath $root -Directory -Force)
        if ($files.Count -eq 0 -and $dirs.Count -eq 1) { $root = $dirs[0].FullName } else { break }
    }
    return $root
}

function Test-PackageContents([string]$Root) {
    $exes = @(Get-ChildItem -LiteralPath $Root -Filter '*.exe' -File)
    if ($exes.Count -eq 0) { throw "The extracted package has no executables at its root ($Root)." }

    $sums = Join-Path $Root 'SHA256SUMS.txt'
    if (-not (Test-Path -LiteralPath $sums -PathType Leaf)) {
        Write-Warn 'The package has no internal SHA256SUMS.txt; skipping extracted-file verification.'
        return
    }
    Write-Note 'Verifying extracted files against SHA256SUMS.txt'
    $map = Read-Sha256Manifest $sums
    foreach ($name in $map.Keys) {
        $path = Join-Path $Root ($name -replace '/', '\')
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Extracted package is missing $name." }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $map[$name]) { throw "Extracted file $name does not match SHA256SUMS.txt." }
    }
}

function Get-BuildInfo([string]$Root) {
    $path = Join-Path $Root 'BUILD-INFO.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    try { return (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json) } catch { return $null }
}

function Remove-Tree([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $Path) { Write-Warn "Could not fully remove $Path; delete it manually." }
    }
}

function Install-Bin([string]$NewRoot, [string]$BinDir) {
    $oldDir = '{0}.old-{1}' -f $BinDir, (Get-Date -Format 'yyyyMMddHHmmss')
    $hadOld = Test-Path -LiteralPath $BinDir
    if ($hadOld) {
        try {
            [IO.Directory]::Move($BinDir, $oldDir)
        } catch {
            throw ("Could not move the current bin folder aside: $($_.Exception.Message)`n" +
                'Stop any running ninfer processes (ninfer-serve.exe, ninfer.exe, ...) and run again.')
        }
    }
    try {
        [IO.Directory]::Move($NewRoot, $BinDir)
    } catch {
        $err = $_.Exception.Message
        if ($hadOld) { [IO.Directory]::Move($oldDir, $BinDir) }
        throw "Could not move the new release into place; the previous bin folder was restored. $err"
    }
    return $oldDir
}

# --- main ---------------------------------------------------------------------------------------

Assert-Platform
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

if (-not $InstallDir) {
    if ($PSScriptRoot) { $InstallDir = $PSScriptRoot } else { $InstallDir = (Get-Location).Path }
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$InstallDir = (Resolve-Path -LiteralPath $InstallDir).Path

$statePath = Join-Path $InstallDir $StateFileName
$binDir = Join-Path $InstallDir 'bin'
$stagingDir = Join-Path $InstallDir '.staging'

$sevenZip = Find-SevenZip
$curl = Find-Curl
Write-Step "Install folder: $InstallDir"
Write-Note "7-Zip: $sevenZip"

Get-ChildItem -LiteralPath $InstallDir -Directory -Filter 'bin.old-*' -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Tree $_.FullName }

$state = Read-State $statePath
$installedVersion = Get-StateValue $state 'version'
$savedFlavor = Get-StateValue $state 'flavor'
if ($savedFlavor -and -not $Flavors.Contains($savedFlavor)) { $savedFlavor = $null }
if (-not (Test-Path -LiteralPath $binDir -PathType Container)) { $installedVersion = $null }

if ($Flavor) {
    $selectedFlavor = $Flavor
} elseif ($savedFlavor) {
    $selectedFlavor = $savedFlavor
} elseif ($Yes -or $CheckOnly) {
    $selectedFlavor = 'dual'
} else {
    $selectedFlavor = Select-Flavor $null
}
$flavorChanged = [bool]($installedVersion -and $selectedFlavor -ne $savedFlavor)
$savedLabel = 'unknown flavor'
if ($savedFlavor) { $savedLabel = $Flavors[$savedFlavor].Label }

if ($installedVersion) {
    Write-Note "Installed: $installedVersion ($savedLabel)"
} else {
    Write-Note 'Installed: none'
}
Write-Note "Flavor:    $($Flavors[$selectedFlavor].Label)"

Write-Step "Checking releases of $Repo"
$release = Get-Release
$package = Resolve-Package $release $selectedFlavor
Write-Note "Release:   $($release.tag_name) (published $($release.published_at))"
Write-Note ("Package:   {0} ({1} parts, {2})" -f $package.Base, $package.Parts.Count, (Format-Size $package.Total))

$cmp = $null
if ($installedVersion) { $cmp = Compare-Tag $installedVersion $release.tag_name }
$needsInstall = $Force -or (-not $installedVersion) -or $flavorChanged -or ($cmp -ne 0)

if (-not $needsInstall) {
    Write-Step "Up to date ($installedVersion, $($Flavors[$selectedFlavor].Label))."
    exit 0
}

if ($installedVersion -and -not $Version -and -not $Force -and -not $flavorChanged -and $cmp -eq -1) {
    Write-Step "Installed $installedVersion is newer than the latest release $($release.tag_name); nothing to do."
    exit 0
}

if (-not $installedVersion) {
    $action = "Install $($release.tag_name)"
} elseif ($flavorChanged) {
    $action = "Switch from $savedLabel $installedVersion to $($Flavors[$selectedFlavor].Label) $($release.tag_name)"
} elseif ($cmp -eq 0) {
    $action = "Reinstall $($release.tag_name)"
} else {
    $action = "Update $installedVersion -> $($release.tag_name)"
}

if ($CheckOnly) {
    Write-Step "Available: $action ($(Format-Size $package.Total) download)."
    exit 2
}

try {
    $drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($InstallDir))
    $needed = [long]($package.Total * 2.5)
    if ($drive.AvailableFreeSpace -lt $needed) {
        Write-Warn ("{0} has {1} free; downloading and extracting this package may need about {2}." -f
            $drive.Name, (Format-Size $drive.AvailableFreeSpace), (Format-Size $needed))
    }
} catch {
    Write-Verbose "Free-space check skipped: $($_.Exception.Message)"
}

if (-not (Confirm-Action "$action ($(Format-Size $package.Total) download)?")) {
    Write-Step 'Cancelled.'
    exit 0
}

$downloadDir = Join-Path $stagingDir 'downloads'
$extractDir = Join-Path $stagingDir 'extract'

Write-Step 'Downloading'
$firstPart = Get-PackageParts $package $downloadDir $curl

Write-Step 'Extracting'
Remove-Tree $extractDir
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
$code = Invoke-Native $sevenZip @('x', $firstPart, "-o$extractDir", '-y', '-bso0', '-bsp2')
if ($code -ne 0) { throw "7-Zip failed with exit code $code." }

$newRoot = Get-PackageRoot $extractDir
Test-PackageContents $newRoot
$buildInfo = Get-BuildInfo $newRoot

Write-Step 'Replacing bin'
$oldDir = Install-Bin $newRoot $binDir

$commit = $package.Commit
if ($buildInfo) {
    foreach ($name in @('source_commit', 'commit', 'revision', 'git_commit')) {
        $p = $buildInfo.PSObject.Properties[$name]
        if ($null -ne $p -and $p.Value) { $commit = [string]$p.Value; break }
    }
}
Write-State $statePath @{
    repo        = $Repo
    flavor      = $selectedFlavor
    version     = $release.tag_name
    commit      = $commit
    package     = $package.Base
    installedAt = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
}

Remove-Tree $oldDir
Remove-Tree $stagingDir

Write-Step "Installed $($release.tag_name) ($($Flavors[$selectedFlavor].Label)) in $binDir"

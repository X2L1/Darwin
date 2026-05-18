[CmdletBinding()]
param(
    [string]$RemoteUrl = "https://github.com/X2L1/Darwin.git",
    [string]$Branch = "training-backups",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $RepoRoot "data\logs"
$LogFile = Join-Path $LogDir "training_backup.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-BackupLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    "[$stamp] $Message" | Tee-Object -FilePath $LogFile -Append
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GitArgs,
        [switch]$AllowFailure
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git @GitArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($output) {
        foreach ($line in $output) {
            Write-BackupLog "git $($GitArgs -join ' '): $line"
        }
    }
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "git $($GitArgs -join ' ') failed with exit code $exitCode"
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output
    }
}

try {
    Set-Location $RepoRoot
    Write-BackupLog "Starting Darwin training backup in $RepoRoot"

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is not installed or is not available on PATH."
    }

    if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
        Invoke-Git @("init") | Out-Null
    }

    Invoke-Git @("checkout", "-B", $Branch) | Out-Null

    $email = (Invoke-Git @("config", "--get", "user.email") -AllowFailure).Output
    if (-not $email) {
        Invoke-Git @("config", "user.email", "darwin-backup@local.invalid") | Out-Null
    }
    $name = (Invoke-Git @("config", "--get", "user.name") -AllowFailure).Output
    if (-not $name) {
        Invoke-Git @("config", "user.name", "Darwin Backup") | Out-Null
    }

    $origin = Invoke-Git @("remote", "get-url", "origin") -AllowFailure
    if ($origin.ExitCode -ne 0) {
        Invoke-Git @("remote", "add", "origin", $RemoteUrl) | Out-Null
    } elseif (($origin.Output | Select-Object -First 1) -ne $RemoteUrl) {
        Write-BackupLog "Origin already exists as '$($origin.Output)'; leaving it unchanged."
    }

    $pathsToAdd = @(
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "configs",
        "darwin",
        "scripts",
        "tests",
        "data\tokenizer.json",
        "data\checkpoints",
        "data\knowledge",
        "data\reviews",
        "data\logs\metrics.jsonl",
        "data\logs\advisory_proposals.jsonl"
    )

    foreach ($path in $pathsToAdd) {
        if (Test-Path (Join-Path $RepoRoot $path)) {
            Invoke-Git @("add", "--", $path) | Out-Null
        }
    }

    $diffCheck = Invoke-Git @("diff", "--cached", "--quiet") -AllowFailure
    $hasStagedChanges = $diffCheck.ExitCode -ne 0
    if ($hasStagedChanges) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
        Invoke-Git @("commit", "-m", "Backup Darwin training state $timestamp") | Out-Null
        Write-BackupLog "Committed training state."
    } else {
        Write-BackupLog "No training-state changes to commit."
    }

    if ($DryRun) {
        Write-BackupLog "Dry run complete; skipping push."
        exit 0
    }

    Invoke-Git @("push", "-u", "origin", "HEAD:refs/heads/$Branch") | Out-Null
    Write-BackupLog "Pushed training state to $RemoteUrl branch $Branch."
    exit 0
} catch {
    Write-BackupLog "Backup failed: $($_.Exception.Message)"
    exit 1
}

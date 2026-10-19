<#
====================================================================
 run_followups.ps1 - runs the four follow-up experiments in order
====================================================================

 Put this in the project root, next to retriever.py, then:

     cd "C:\Users\h\OneDrive\Desktop\New folder (5)\Medical_RAG_Project"
     powershell -ExecutionPolicy Bypass -File .\run_followups.ps1

 The -ExecutionPolicy Bypass is needed because unsigned .ps1 files are
 blocked by default. It applies to this one invocation only.

 EVERYTHING IS RESUMABLE. If the power goes out or you close the window,
 run the exact same command again and each script picks up from its last
 checkpoint. Nothing is recomputed.

 WHAT IT DOES BEFORE RUNNING ANYTHING
   - pauses OneDrive (it has competed for RAM during long runs here)
   - stops the machine sleeping mid-run
   - checks ollama is up and the model answers
   - checks python packages, installs scikit-learn if missing
   - refuses to start if the results folder is missing

 FLAGS
   -Only a2|sweep|gate|phase2     run one step instead of all four
   -SkipPhase2                    skip the slowest step (default: on)
   -KeepAwake:$false              allow sleep (not recommended)

 EXAMPLES
   powershell -ExecutionPolicy Bypass -File .\run_followups.ps1
   powershell -ExecutionPolicy Bypass -File .\run_followups.ps1 -Only a2
   powershell -ExecutionPolicy Bypass -File .\run_followups.ps1 -Only sweep
#>

[CmdletBinding()]
param(
    [ValidateSet("all", "a2", "sweep", "gate", "phase2")]
    [string]$Only = "all",
    [switch]$SkipPhase2 = $true,
    [bool]$KeepAwake = $true
)

# Continue, not Stop: native stderr merged with 2>&1 would otherwise
# throw on ordinary python progress output. Exit codes are checked
# explicitly everywhere below.
$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# ------------------------------------------------------------------ logging
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "followups_$Stamp.log"

function Say {
    param([string]$Text, [string]$Colour = "Gray")
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Text
    Write-Host $line -ForegroundColor $Colour
    Add-Content -Path $LogFile -Value $line
}

function Header {
    param([string]$Text)
    $bar = "=" * 68
    Write-Host ""
    Write-Host $bar -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host $bar -ForegroundColor Cyan
    Add-Content -Path $LogFile -Value "`n$bar`n$Text`n$bar"
}

Header "Medical RAG - follow-up experiments"
Say "project : $ProjectRoot"
Say "log     : $LogFile"

# ------------------------------------------------------- virtual environment
$Activate = Join-Path $ProjectRoot "rag_env\Scripts\Activate.ps1"
if (Test-Path $Activate) {
    Say "activating rag_env" "Green"
    . $Activate      # dot-sourced: & would activate in a child scope and vanish
} else {
    Say "rag_env not found at $Activate - using whatever python is on PATH" "Yellow"
}

$PyVer = (python -c "import sys; print(sys.version.split()[0])" 2>&1)
Say "python  : $PyVer"

# ------------------------------------------------------------- sanity checks
Header "Pre-run checks"

if (-not (Test-Path (Join-Path $ProjectRoot "evaluation\results"))) {
    Say "FAILED: evaluation\results not found. Run the main study first." "Red"
    exit 1
}
Say "results folder present" "Green"

foreach ($f in @("run_a2_control.py", "run_corpus_sweep.py",
                 "train_selective_gate.py", "rerun_phase2_greedy.py")) {
    if (-not (Test-Path (Join-Path $ProjectRoot $f))) {
        Say "FAILED: $f missing from the project root" "Red"
        exit 1
    }
}
Say "all four scripts present" "Green"

# required per-question outcome files
$needed = @("full_medqa_A.csv", "full_medqa_C.csv",
            "full_medmcqa_A.csv", "full_medmcqa_C.csv")
foreach ($n in $needed) {
    if (-not (Test-Path (Join-Path $ProjectRoot "evaluation\results\$n"))) {
        Say "WARNING: evaluation\results\$n missing - some comparisons will be skipped" "Yellow"
    }
}

# ollama
Say "checking ollama ..."
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 10
    $names = $tags.models | ForEach-Object { $_.name }
    Say ("ollama up, {0} models loaded" -f $names.Count) "Green"
    if ($names -notcontains "qwen2.5:7b") {
        Say "WARNING: qwen2.5:7b not in the model list. Run: ollama pull qwen2.5:7b" "Yellow"
    }
} catch {
    Say "FAILED: ollama is not responding on port 11434." "Red"
    Say "Open another window and run:  ollama serve" "Red"
    exit 1
}

# python packages
$pkgs = @{ "pandas" = "pandas"; "numpy" = "numpy"; "scipy" = "scipy";
           "faiss"  = "faiss-cpu"; "sklearn" = "scikit-learn" }
foreach ($mod in $pkgs.Keys) {
    python -c "import $mod" 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($mod -eq "sklearn") {
            Say "scikit-learn missing - installing (needed only by the gate step)" "Yellow"
            python -m pip install --quiet scikit-learn
            if ($LASTEXITCODE -ne 0) { Say "pip install failed; the gate step will be skipped" "Yellow" }
        } else {
            Say "FAILED: python module '$mod' not importable (pip install $($pkgs[$mod]))" "Red"
            exit 1
        }
    }
}
Say "python packages ok" "Green"

# ---------------------------------------------------------------- housekeeping
Header "Housekeeping"

$od = Get-Process -Name "OneDrive" -ErrorAction SilentlyContinue
if ($od) {
    Stop-Process -Name "OneDrive" -Force -ErrorAction SilentlyContinue
    Say "OneDrive stopped (it will restart at next login)" "Green"
} else {
    Say "OneDrive not running" "Green"
}

if ($KeepAwake) {
    powercfg /change standby-timeout-ac 0    | Out-Null
    powercfg /change hibernate-timeout-ac 0  | Out-Null
    powercfg /change monitor-timeout-ac 20   | Out-Null
    Say "sleep disabled for the duration (monitor still sleeps at 20 min)" "Green"
}

try {
    $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null
    if ($gpu) { Say "gpu     : $gpu" }
} catch { }

# ------------------------------------------------------------------- runner
function Invoke-Step {
    param([string]$Name, [string]$Script, [string[]]$ScriptArgs = @(), [string]$Note = "")

    Header "$Name"
    if ($Note) { Say $Note }

    $cmd = (@("python", $Script) + $ScriptArgs) -join " "
    Say "running: $cmd" "White"
    $t0 = Get-Date

    # Tee so you see progress live AND get a full transcript on disk.
    & python $Script @ScriptArgs 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
    $code = $LASTEXITCODE

    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0) {
        Say "$Name finished in $mins min" "Green"
    } else {
        Say "$Name exited with code $code after $mins min" "Red"
        Say "Re-run this same command to resume from the last checkpoint." "Yellow"
    }
    return $code
}

$failed = @()

# 1 --- A2 prompt control (do this one first)
if ($Only -in @("all", "a2")) {
    $rc = Invoke-Step -Name "STEP 1/4  A2 prompt control" `
                      -Script "run_a2_control.py" `
                      -Note   "~3-4 hours. Closes the biggest objection to the paper."
    if ($rc -ne 0) { $failed += "a2" }
}

# 2 --- corpus coverage sweep
if ($Only -in @("all", "sweep")) {
    $rc = Invoke-Step -Name "STEP 2/4  Corpus-coverage sweep (MedQA)" `
                      -Script "run_corpus_sweep.py" `
                      -Note   "~4 hours. Add --both later for MedMCQA (~15 h more)."
    if ($rc -ne 0) { $failed += "sweep" }
}

# 3 --- learned gate: dump features, then train
if ($Only -in @("all", "gate")) {
    $rc = Invoke-Step -Name "STEP 3a/4  Gate features" `
                      -Script "train_selective_gate.py" -ScriptArgs @("--dump") `
                      -Note   "~40 min. Retrieval only, no generation."
    if ($rc -ne 0) {
        $failed += "gate-dump"
    } else {
        $rc = Invoke-Step -Name "STEP 3b/4  Gate training" `
                          -Script "train_selective_gate.py" -ScriptArgs @("--train") `
                          -Note "Seconds. CPU only."
        if ($rc -ne 0) { $failed += "gate-train" }
    }
}

# 4 --- phase 2 re-run (slow; off by default)
if ($Only -eq "phase2" -or ($Only -eq "all" -and -not $SkipPhase2)) {
    $rc = Invoke-Step -Name "STEP 4/4  Phase 2 under greedy decoding" `
                      -Script "rerun_phase2_greedy.py" -ScriptArgs @("--limit", "500") `
                      -Note   "~5 hours with --limit 500."
    if ($rc -ne 0) { $failed += "phase2" }
} elseif ($Only -eq "all") {
    Header "STEP 4/4  Phase 2 re-run - SKIPPED"
    Say "Run it separately when you have ~5 hours spare:" "Yellow"
    Say "  powershell -ExecutionPolicy Bypass -File .\run_followups.ps1 -Only phase2" "Yellow"
}

# ------------------------------------------------------------------ summary
Header "Summary"

$outputs = @(
    "evaluation\results\a2_control_summary.csv",
    "evaluation\results\corpus_sweep_summary.csv",
    "evaluation\results\learned_gate_summary.csv",
    "evaluation\results\phase2_greedy_summary.csv"
)
foreach ($o in $outputs) {
    $p = Join-Path $ProjectRoot $o
    if (Test-Path $p) {
        Say "WROTE  $o" "Green"
    } else {
        Say "absent $o" "DarkGray"
    }
}

if ($failed.Count -gt 0) {
    Say ("steps that did not finish: {0}" -f ($failed -join ", ")) "Yellow"
    Say "Re-run the same command - everything resumes." "Yellow"
} else {
    Say "all requested steps completed" "Green"
}

if ($KeepAwake) {
    Say "NOTE: sleep settings were changed and do NOT revert on their own." "Yellow"
    Say "  restore with:  powercfg /change standby-timeout-ac 30" "Yellow"
}

Say "full transcript: $LogFile"
Write-Host ""
Write-Host "Next: read a2_control_summary.csv first. If A2 vs A is not" -ForegroundColor Cyan
Write-Host "significant, the paper's Finding 4 stands as written." -ForegroundColor Cyan
Write-Host ""

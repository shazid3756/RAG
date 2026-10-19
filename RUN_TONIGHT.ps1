<#
    RUN_TONIGHT.ps1 - the two remaining GPU jobs, in order, unattended.

    cd "C:\Users\h\OneDrive\Desktop\New folder (5)\Medical_RAG_Project"
    .\RUN_TONIGHT.ps1

    JOB 1  MedMCQA coverage sweep          ~14 h   4,183 items x 4 levels
    JOB 2  Seed variance at 500k, 3 seeds  ~3 h    1,273 items x 3 draws

    Job 1 first because it is the long one and because its result changes a
    claim in the paper: inverse corpus scaling is currently measured only on
    MedQA, which is the WEAKER of the two benchmarks. MedMCQA is where the
    retrieval penalty is decisive (-5.71 pp, survives Holm), so a reviewer
    will ask why the named effect was never measured there.

    Job 2 answers a different question. At 4.18% coverage the penalty
    recovered by +2.67 pp, and we cannot currently tell whether that is the
    curve turning or just which passages happened to land in the index.
    Redrawing the SAME coverage level with different seeds measures the
    corpus-sampling spread directly. Seed 42 is the draw already reported, so
    seeds 7, 13 and 99 give four draws in total.

    Both jobs checkpoint every 20 items and resume where they stopped, so a
    power cut costs minutes, not hours. Re-running this script after an
    interruption is the intended recovery: every step re-checks its marker
    file and skips only if the row count is actually complete.
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log   = Join-Path $Root "run_tonight_$Stamp.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $Log -Value $line
}

function Invoke-Py {
    <#
      Run python, stream stdout+stderr to console and log, return exit code.

      Two traps, both load-bearing:

      1. $ErrorActionPreference. When a native command's stderr is redirected
         with 2>&1, PowerShell wraps each stderr line in an ErrorRecord, and
         under "Stop" the FIRST one becomes a terminating NativeCommandError.
         torch, transformers and faiss all print load-time notices to stderr,
         so with "Stop" in force this script would die in its first minute on
         a warning that means nothing. Relax it here, restore it after.

      2. Out-Host at the end of the pipeline. Tee-Object PASSES ITS INPUT
         THROUGH, so without Out-Host every line python printed would fall
         into the function's output stream and be returned alongside the exit
         code. The caller's `$code = Invoke-Py ...` would then hold an array
         of thousands of strings, `$code -ne 0` would evaluate as a filter
         rather than a comparison, and the script would report FAILED after a
         run that succeeded perfectly.
    #>
    param([Parameter(Mandatory)][string[]]$PyArgs)

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @PyArgs 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Rows($path) {
    if (-not (Test-Path $path)) { return 0 }
    try { return @(Import-Csv -Path $path).Count } catch { return 0 }
}

# ------------------------------------------------------------- preflight
# Fail in ten seconds on a missing input rather than thirteen hours later,
# when job 2 finally reaches for an index path that was never there.
$Needed = @{
    "python on PATH"        = $null
    "evaluation\medqa_test.json"                  = "job 2"
    "evaluation\medmcqa_val.json"                 = "job 1"
    "evaluation\results\full_medqa_A.csv"         = "job 2 baseline"
    "evaluation\results\full_medmcqa_A.csv"       = "job 1 baseline"
    "run_corpus_sweep.py"                         = "both"
    "data\large\pubmed_1m_chunks.csv"             = "job 2 corpus"
    "vector_db_large\minilm_index.faiss"          = "job 2 index"
}

$missing = @()
foreach ($k in $Needed.Keys) {
    if ($k -eq "python on PATH") {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            $missing += "python is not on PATH (activate rag_env first)"
        }
        continue
    }
    if (-not (Test-Path (Join-Path $Root $k))) {
        $missing += "$k  (needed by $($Needed[$k]))"
    }
}
if ($missing.Count) {
    Say "PREFLIGHT FAILED - nothing was run:"
    $missing | ForEach-Object { Say "  missing: $_" }
    exit 1
}
Say "Preflight OK. Log: $Log"

# OneDrive syncing multi-GB indexes while a 7B model is resident has caused
# RAM pressure on this machine before. It restarts at next login.
Say "Stopping OneDrive (frees RAM; restarts at next login)"
Stop-Process -Name "OneDrive" -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- job 1
$McqaRows = Rows (Join-Path $Root "evaluation\results\sweep_medmcqa_100000.csv")
if ($McqaRows -ge 4183) {
    Say "JOB 1 already complete ($McqaRows rows) - skipping"
} else {
    Say "JOB 1  MedMCQA coverage sweep, 4 levels (~14 h). Resuming if partial."
    # --datasets medmcqa, not --both: the MedQA levels under this tag are
    # already finished and would resume as no-ops, but naming the dataset
    # says so out loud instead of relying on that.
    $code = Invoke-Py @("run_corpus_sweep.py", "--datasets", "medmcqa",
                        "--levels", "10000", "25000", "50000", "100000",
                        "--tag", "sweep")
    if ($code -ne 0) { Say "JOB 1 FAILED (exit $code) - stopping"; exit 1 }
    Say "JOB 1 done"
}

# ---------------------------------------------------------------- job 2
foreach ($s in 7, 13, 99) {
    # ${s} braces, not $s_: PowerShell would read "$s_medqa" as a variable
    # named s_medqa and silently build the path "evaluation\results\seedvar".
    $marker = Join-Path $Root "evaluation\results\seedvar${s}_medqa_500000.csv"
    $r = Rows $marker
    if ($r -ge 1273) {
        Say "JOB 2 seed $s already complete ($r rows) - skipping"
        continue
    }
    Say "JOB 2  seed $s - 500k level, fresh subset draw (~1 h)"
    $code = Invoke-Py @("run_corpus_sweep.py",
                        "--chunks-file", "data\large\pubmed_1m_chunks.csv",
                        "--index-file", "vector_db_large\minilm_index.faiss",
                        "--total-abstracts", "1000000",
                        "--levels", "500000",
                        "--seed", "$s", "--tag", "seedvar$s")
    if ($code -ne 0) { Say "JOB 2 seed $s FAILED (exit $code) - stopping"; exit 1 }
    Say "JOB 2 seed $s done"
}

Say "ALL DONE."
Say "Expect:"
Say "  evaluation\results\sweep_medmcqa_{10000,25000,50000,100000}.csv"
Say "  evaluation\results\sweep_summary.csv"
Say "  evaluation\results\seedvar{7,13,99}_medqa_500000.csv"
Say "  evaluation\results\seedvar{7,13,99}_summary.csv"

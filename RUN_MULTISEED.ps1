<#
    RUN_MULTISEED.ps1 - three extra corpus draws at every sweep level.

    cd "C:\Users\h\OneDrive\Desktop\New folder (5)\Medical_RAG_Project"
    .\RUN_MULTISEED.ps1                 # JOB A only,  ~13 h  (default)
    .\RUN_MULTISEED.ps1 -Jobs B         # JOB B only,  ~10 h
    .\RUN_MULTISEED.ps1 -Jobs AB        # both,        ~23 h

    WHY
    ---
    Four draws at one level (2.09% coverage) gave -5.11, -5.26, -3.46 and
    -1.81: an SD of 1.62 pp and a range of 3.46 pp from nothing but a
    different random subset of the same passages at the same coverage. That
    spread is wider than the whole coverage trend the paper reports across a
    decade of corpus growth, and wider than the +2.67 pp "recovery" at the top
    level. So every sweep level needs several draws, not one, before the slope
    can be quoted as a result.

    The bootstrap intervals already in the paper resample QUESTIONS. They are
    the wrong reference for a claim about corpus size: two labs repeating this
    study would index different passages, not quiz different students.

    JOB A  100k corpus, MedQA, 4 levels x 3 new seeds  = 15,276 generations
    JOB B  1M corpus,  MedQA, 4 levels x 3 new seeds, minus the 500k level
           which already exists under these tags        = 11,457 generations

    Job B reuses the seedvar7/13/99 tags deliberately, so its 500k level
    resumes as a no-op rather than costing another three hours.

    Both jobs checkpoint every 20 items and resume. Re-running after an
    interruption is the intended recovery.
#>

param(
    # A = extra draws on the 100k corpus (the sweep whose slope the paper
    #     names). B = the same on the 1M corpus. AB = both, A first.
    [ValidateSet('A', 'B', 'AB')]
    [string]$Jobs = 'A',

    # Kept so an already-typed command still works; it just means -Jobs AB.
    [switch]$IncludeUpsweep
)

if ($IncludeUpsweep -and $Jobs -eq 'A') { $Jobs = 'AB' }
$RunA = $Jobs.Contains('A')
$RunB = $Jobs.Contains('B')

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log   = Join-Path $Root "run_multiseed_$Stamp.log"
$Seeds = 7, 13, 99

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $Log -Value $line
}

function Invoke-Py {
    <#
      Three things this wrapper exists for, all learned the hard way:

      1. python -u. Without it, stdout through a pipe is block-buffered and
         the progress line (which ends in \r, not \n) never flushes, so a
         perfectly healthy run looks frozen for an hour at a time.

      2. $ErrorActionPreference. Native stderr redirected with 2>&1 becomes
         ErrorRecords, and under "Stop" the first one is a terminating
         NativeCommandError. torch and faiss print notices to stderr at load,
         so "Stop" would kill the run on a harmless warning.

      3. Out-Host at the end. Tee-Object passes its input through, so without
         it every printed line would join the function's output stream and
         the caller's exit-code check would compare against an array of
         thousands of strings instead of an integer.
    #>
    param([Parameter(Mandatory)][string[]]$PyArgs)

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -u @PyArgs 2>&1 | Tee-Object -FilePath $Log -Append | Out-Host
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
$need = @(
    "run_corpus_sweep.py",
    "summarise_draws.py",
    "evaluation\medqa_test.json",
    "evaluation\results\full_medqa_A.csv"
)
if ($RunB) {
    $need += "data\large\pubmed_1m_chunks.csv"
    $need += "vector_db_large\minilm_index.faiss"
}
$missing = @()
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $missing += "python is not on PATH (activate rag_env first)"
}
foreach ($f in $need) {
    if (-not (Test-Path (Join-Path $Root $f))) { $missing += $f }
}
if ($missing.Count) {
    Say "PREFLIGHT FAILED - nothing was run:"
    $missing | ForEach-Object { Say "  missing: $_" }
    exit 1
}
Say "Preflight OK. Running job(s): $Jobs. Log: $Log"
Say "Stopping OneDrive (frees RAM; restarts at next login)"
Stop-Process -Name "OneDrive" -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- job A
if ($RunA) {
    Say "JOB A  100k corpus, MedQA, seeds $($Seeds -join ', ') across 4 levels (~13 h)"
    foreach ($s in $Seeds) {
        # ${s} braces: "$s_medqa" would parse as a variable named s_medqa and
        # silently collapse the path.
        $marker = Join-Path $Root "evaluation\results\draw${s}_medqa_100000.csv"
        $r = Rows $marker
        if ($r -ge 1273) {
            Say "  seed $s already complete ($r rows at the top level) - skipping"
            continue
        }
        Say "  seed $s starting (4 levels, ~4.3 h)"
        $code = Invoke-Py @("run_corpus_sweep.py",
                            "--levels", "10000", "25000", "50000", "100000",
                            "--seed", "$s", "--tag", "draw$s")
        if ($code -ne 0) { Say "JOB A seed $s FAILED (exit $code) - stopping"; exit 1 }
        Say "  seed $s done"
    }
    Say "JOB A done"
} else {
    Say "JOB A skipped (-Jobs $Jobs)"
}

# ---------------------------------------------------------------- job B
if ($RunB) {
    Say "JOB B  1M corpus, MedQA, same seeds across 4 levels (~10 h)"
    foreach ($s in $Seeds) {
        $marker = Join-Path $Root "evaluation\results\seedvar${s}_medqa_1000000.csv"
        $r = Rows $marker
        if ($r -ge 1273) {
            Say "  seed $s already complete ($r rows at the top level) - skipping"
            continue
        }
        Say "  seed $s starting (500k level already exists, so 3 new levels)"
        $code = Invoke-Py @("run_corpus_sweep.py",
                            "--chunks-file", "data\large\pubmed_1m_chunks.csv",
                            "--index-file", "vector_db_large\minilm_index.faiss",
                            "--total-abstracts", "1000000",
                            "--levels", "100000", "250000", "500000", "1000000",
                            "--seed", "$s", "--tag", "seedvar$s")
        if ($code -ne 0) { Say "JOB B seed $s FAILED (exit $code) - stopping"; exit 1 }
        Say "  seed $s done"
    }
    Say "JOB B done"
} else {
    Say "JOB B skipped (-Jobs $Jobs)"
}

# --------------------------------------------------------------- verdict
Say "Summarising"
[void](Invoke-Py @("summarise_draws.py"))
Say "ALL DONE. Log: $Log"

# =====================================================================
#  RUN_ALL.ps1 - everything still outstanding, in return-on-time order
#  ---------------------------------------------------------------------
#  Run from the project root:
#
#      cd "C:\Users\h\OneDrive\Desktop\New folder (5)\Medical_RAG_Project"
#      .\RUN_ALL.ps1                  # everything, in order
#      .\RUN_ALL.ps1 -Only sweep      # just one step
#      .\RUN_ALL.ps1 -List            # show steps and status, run nothing
#
#  Every step is resumable. Ctrl+C is safe. A power cut is safe. Re-running
#  a finished step costs seconds, not hours, because each underlying script
#  checks its own output first and skips completed work.
#
#  Total GPU time if you run the lot: roughly 16 hours. You do not have to
#  do it in one sitting - that is the whole point of the resume logic.
# =====================================================================

[CmdletBinding()]
param(
    [string[]] $Only,
    [switch]   $List,
    [switch]   $SkipOneDrive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# ---------------------------------------------------------------- venv
$Activate = Join-Path $ProjectRoot "rag_env\Scripts\Activate.ps1"
if (Test-Path $Activate) {
    # Dot-sourced, NOT '&'. Calling it with '&' activates the venv in a child
    # scope that dies immediately, so every python call afterwards silently
    # uses the system interpreter and fails on missing packages.
    . $Activate
    Write-Host "venv active: $env:VIRTUAL_ENV" -ForegroundColor DarkGray
} else {
    Write-Warning "rag_env not found at $Activate - using system python"
}

# ---------------------------------------------------------------- steps
# marker = a file whose existence means this step is already done.
$Steps = @(
    @{ Key="sweep";   Order=1
       Name="Corpus-coverage sweep (MedQA, 4 levels)"
       Why ="Earns the title. Turns 'retrieval hurts at 0.4%' into a curve."
       Cmd ="python run_corpus_sweep.py"
       Marker="evaluation\results\corpus_sweep_summary.csv"
       MinRows=4
       Hours="~4" }

    @{ Key="a2";      Order=2
       Name="A2 prompt control"
       Why ="Removes the one confound in the central contrast."
       Cmd ="python run_a2_control.py"
       Marker="evaluation\results\full_medmcqa_A2.csv"
       MinRows=4183
       Hours="~3-4" }

    @{ Key="rctd";    Order=3
       Name="Stage D over the 20k RCT corpus"
       Why ="Completes the cross-corpus replication you already started."
       Cmd ="python run_rct_corpus.py --eval --finetuned"
       Marker="evaluation\results\rct_medmcqa_D.csv"
       MinRows=4183
       Hours="~4" }

    @{ Key="phase2";  Order=4
       Name="Phase 2 re-run under greedy decoding"
       Why ="Removes the two-protocol inconsistency."
       Cmd ="python rerun_phase2_greedy.py --limit 500"
       Marker="evaluation\results\phase2_greedy_summary.csv"
       MinRows=10
       Hours="~5" }

    @{ Key="gate";    Order=5
       Name="Learned selective gate (optional)"
       Why ="Tests whether a trained gate can clear s*. Cheap, CPU only."
       Cmd ="python train_selective_gate.py --dump; python train_selective_gate.py --train"
       Marker="evaluation\results\learned_gate_summary.csv"
       MinRows=8
       Hours="~0.2" }

    # The downward sweep (step 1) can only show the penalty deepening as the
    # corpus grows; it cannot show where the curve turns, because the subset
    # trick can never index more than it was given. collect_large_corpus.py
    # has since built and indexed 1,157,180 chunks over 1,000,000 abstracts,
    # so the same machinery now runs UPWARD from the same 100k join point.
    # The 100,000 level here is a fresh sample from a different corpus, so it
    # doubles as a replication of step 1's endpoint.
    @{ Key="upsweep"; Order=6
       Name="Upward corpus sweep to 1M abstracts (MedQA, 4 levels)"
       Why ="Turns the paper's biggest stated limitation into a result."
       Cmd ="python run_corpus_sweep.py --chunks-file data\large\pubmed_1m_chunks.csv --index-file vector_db_large\minilm_index.faiss --total-abstracts 1000000 --levels 100000 250000 500000 1000000 --tag upsweep"
       Marker="evaluation\results\upsweep_summary.csv"
       MinRows=4
       Hours="~5" }
)

function Test-StepDone {
    <#
      A step is done when its marker exists AND holds the expected number of
      data rows. Existence alone is not enough: every underlying script
      checkpoints as it goes, so an interrupted run leaves a real but partial
      CSV behind. The first version of this script tested existence only, and
      would have skipped two runs that had stopped at 10% and 40%.
    #>
    param($Step)

    $path = Join-Path $ProjectRoot $Step.Marker
    if (-not (Test-Path $path)) { return @{ Done=$false; Rows=0 } }

    # Count data rows properly: some CSVs carry a raw_on_failure column whose
    # text contains embedded newlines, so a plain line count over-reports.
    $rows = 0
    try {
        $rows = @(Import-Csv -Path $path).Count
    } catch {
        $rows = 0
    }

    $need = if ($Step.MinRows) { [int]$Step.MinRows } else { 1 }
    return @{ Done = ($rows -ge $need); Rows = $rows; Need = $need }
}

function Show-Status {
    Write-Host ""
    Write-Host ("{0,-3} {1,-46} {2,-8} {3}" -f "#","Step","GPU","Status")
    Write-Host ("-" * 92)
    foreach ($s in $Steps | Sort-Object Order) {
        $st   = Test-StepDone $s
        $mark = if ($st.Done) { "DONE" }
                elseif ($st.Rows -gt 0) { "PARTIAL {0}/{1}" -f $st.Rows, $st.Need }
                else { "pending" }
        $col  = if ($st.Done) { "Green" } elseif ($st.Rows -gt 0) { "Magenta" } else { "Yellow" }
        Write-Host ("{0,-3} {1,-46} {2,-8} " -f $s.Order, $s.Name, $s.Hours) -NoNewline
        Write-Host $mark -ForegroundColor $col
        Write-Host ("    {0}" -f $s.Why) -ForegroundColor DarkGray
    }
    Write-Host ""
}

if ($List) { Show-Status; exit 0 }

# ---------------------------------------------------------------- OneDrive
# OneDrive sync competes for the 16GB of RAM these runs need and has caused
# real failures on this machine. Stopped, not disabled - it restarts on next
# login or when you launch it.
if (-not $SkipOneDrive) {
    Stop-Process -Name "OneDrive" -Force -ErrorAction SilentlyContinue
    Write-Host "OneDrive stopped for the duration of the run." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- preflight
Write-Host "`nChecking Ollama is up..." -ForegroundColor Cyan
try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
    $names = $tags.models.name
    Write-Host ("  {0} models loaded" -f $names.Count) -ForegroundColor Green
    foreach ($need in @("qwen2.5:7b", "qwen2.5-medqa-lora")) {
        if ($names -notcontains $need) {
            Write-Warning "  '$need' is NOT registered - steps needing it will fail"
        }
    }
} catch {
    Write-Host "  Ollama is not responding on 127.0.0.1:11434." -ForegroundColor Red
    Write-Host "  Start it (run 'ollama serve' or open the Ollama app) and retry." -ForegroundColor Red
    exit 1
}

Show-Status

# ---------------------------------------------------------------- run
$ToRun = $Steps | Sort-Object Order
if ($Only) { $ToRun = $ToRun | Where-Object { $Only -contains $_.Key } }

$Log = Join-Path $ProjectRoot ("run_all_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Logging to $Log`n" -ForegroundColor DarkGray

foreach ($s in $ToRun) {

    $st = Test-StepDone $s
    if ($st.Done) {
        Write-Host ("[{0}] {1} - already complete, skipping" -f $s.Order, $s.Name) -ForegroundColor DarkGray
        continue
    }
    if ($st.Rows -gt 0) {
        Write-Host ("[{0}] {1} - PARTIAL ({2}/{3}), resuming" -f `
                    $s.Order, $s.Name, $st.Rows, $st.Need) -ForegroundColor Magenta
    }

    Write-Host ("`n" + ("=" * 78)) -ForegroundColor Cyan
    Write-Host ("[{0}] {1}   ({2} h)" -f $s.Order, $s.Name, $s.Hours) -ForegroundColor Cyan
    Write-Host ("      {0}" -f $s.Why) -ForegroundColor DarkGray
    Write-Host ("=" * 78) -ForegroundColor Cyan

    $started = Get-Date
    Add-Content $Log ("`n=== {0} started {1}" -f $s.Name, $started)

    # Each command runs in this shell so the venv applies. Failures are
    # caught per-step: one broken step must not abandon the ones after it.
    try {
        Invoke-Expression $s.Cmd
        $code = $LASTEXITCODE
    } catch {
        Write-Warning ("  step threw: {0}" -f $_.Exception.Message)
        $code = 1
    }

    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

    $after = Test-StepDone $s
    if ($code -eq 0 -and $after.Done) {
        Write-Host ("  OK - {0} min" -f $mins) -ForegroundColor Green
        Add-Content $Log ("=== {0} OK in {1} min" -f $s.Name, $mins)
    } else {
        Write-Warning ("  INCOMPLETE: {0}/{1} rows after {2} min (exit {3}). Re-run to resume." -f `
                       $after.Rows, $after.Need, $mins, $code)
        Add-Content $Log ("=== {0} INCOMPLETE {1}/{2} exit {3}" -f $s.Name, $after.Rows, $after.Need, $code)
    }
}

# ---------------------------------------------------------------- analysis
Write-Host ("`n" + ("=" * 78)) -ForegroundColor Cyan
Write-Host "ANALYSIS - no GPU, seconds" -ForegroundColor Cyan
Write-Host ("=" * 78) -ForegroundColor Cyan

python run_robustness_stats.py
python run_rct_analysis.py
python finalise_gpu_sections.py --write

Write-Host "`nDone." -ForegroundColor Green
Show-Status
Write-Host "New/updated paper sections:" -ForegroundColor Cyan
Get-ChildItem -Path $ProjectRoot -Filter "1*_*.tex" -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host ("  " + $_.Name) }
Write-Host "  paper\sections\09_statistical_robustness.tex"
Write-Host ""
Write-Host "Restart OneDrive when convenient:  Start-Process `"$env:LOCALAPPDATA\Microsoft\OneDrive\OneDrive.exe`"" -ForegroundColor DarkGray

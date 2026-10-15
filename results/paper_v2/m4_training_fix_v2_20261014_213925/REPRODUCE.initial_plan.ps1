$studyStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$studyOutput = Join-Path 'C:\antigravity_projects\dcsnew' ('results\paper_v2\m4_training_fix_v2_' + $studyStamp + '_reproduce')
New-Item -ItemType Directory -Path $studyOutput -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'C:\antigravity_projects\dcsnew\results\paper_v2\m4_training_fix_v2_20261014_213925\claim.md' -Destination (Join-Path $studyOutput 'claim.md') -ErrorAction Stop
[pscustomobject]@{output_dir=$studyOutput; started_at_utc=(Get-Date).ToUniversalTime().ToString('o');budget_hours=9} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studyOutput 'session.json')
& 'C:\Users\ragha\AppData\Local\Programs\Python\Python311\python.exe' 'C:\antigravity_projects\dcsnew\pipeline-crack-gnn\paper_v2\m4_training_fix_v2.py' run --output $studyOutput
if ($LASTEXITCODE -ne 0) { throw 'Study stopped; inspect RESULTS.md and failures.jsonl. Do not retry a consumed TITS pass.' }

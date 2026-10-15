$auditStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$auditOutput = Join-Path 'C:\antigravity_projects\dcsnew' ('results\paper_v2\m4_training_fix_' + $auditStamp + '_audit_repro')
& 'C:\Users\ragha\AppData\Local\Programs\Python\Python311\python.exe' 'C:\antigravity_projects\dcsnew\results\paper_v2\m4_training_fix_20261014_212451\audit_partial.py' --output $auditOutput
if ($LASTEXITCODE -ne 0) { throw 'Partial audit reproduction failed' }

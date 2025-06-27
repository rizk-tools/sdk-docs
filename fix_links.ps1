# PowerShell script to fix old numbered directory references
$replacements = @{
    "../01-getting-started/" = "../getting-started/"
    "../02-core-concepts/" = "../core-concepts/"
    "../03-framework-integration/" = "../framework-integration/"
    "../04-decorators/" = "../decorators/"
    "../05-llm-adapters/" = "../llm-adapters/"
    "../06-guardrails/" = "../guardrails/"
    "../07-observability/" = "../observability/"
    "../08-advanced-config/" = "../advanced-config/"
    "../09-api-reference/" = "../api-reference/"
    "../11-troubleshooting/" = "../troubleshooting/"
}

# Get all markdown files in the docs directory (excluding hidden directories)
$files = Get-ChildItem -Path "src/content/docs" -Filter "*.md" -Recurse | Where-Object { $_.FullName -notmatch "[\\/]_" }

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    foreach ($oldPath in $replacements.Keys) {
        $newPath = $replacements[$oldPath]
        $content = $content -replace [regex]::Escape($oldPath), $newPath
    }
    
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated: $($file.FullName)"
    }
}

Write-Host "Link fixing completed!" 
# PowerShell script to fix all numbered directory references
$replacements = @{
    # Directory references with trailing slash
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
    
    # File references with .md extension
    "../01-getting-started/([^/]+)\.md" = "../getting-started/`$1/"
    "../02-core-concepts/([^/]+)\.md" = "../core-concepts/`$1/"
    "../03-framework-integration/([^/]+)\.md" = "../framework-integration/`$1/"
    "../04-decorators/([^/]+)\.md" = "../decorators/`$1/"
    "../05-llm-adapters/([^/]+)\.md" = "../llm-adapters/`$1/"
    "../06-guardrails/([^/]+)\.md" = "../guardrails/`$1/"
    "../07-observability/([^/]+)\.md" = "../observability/`$1/"
    "../08-advanced-config/([^/]+)\.md" = "../advanced-config/`$1/"
    "../09-api-reference/([^/]+)\.md" = "../api-reference/`$1/"
    "../11-troubleshooting/([^/]+)\.md" = "../troubleshooting/`$1/"
}

# Get all markdown files in the docs directory (excluding hidden directories)
$files = Get-ChildItem -Path "src/content/docs" -Filter "*.md" -Recurse | Where-Object { $_.FullName -notmatch "[\\/]_" }

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    foreach ($pattern in $replacements.Keys) {
        $replacement = $replacements[$pattern]
        $content = $content -replace $pattern, $replacement
    }
    
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated: $($file.FullName)"
    }
}

Write-Host "Comprehensive link fixing completed!" 
# TaskForce 桌面应用构建脚本(Windows PowerShell)
#
# 用法(在仓库根或任意位置):
#   powershell -ExecutionPolicy Bypass -File packaging/build.ps1
#   powershell -ExecutionPolicy Bypass -File packaging/build.ps1 -SkipFront     # 前端产物已就绪
#   powershell -ExecutionPolicy Bypass -File packaging/build.ps1 -Only debug    # 只出 console 变体
#
# 产物:
#   dist/release/TaskForce/TaskForce.exe            交付用(windowed,双击即用)
#   dist/debug/TaskForce-debug/TaskForce-debug.exe  排障用(带控制台,--tf-smoke 自检也用它)
#
# ⚠️ 交付时**整个文件夹一起给**:只拷 exe 必然失败,同级 _internal/ 里是全部依赖。
# ⚠️ 为什么产两个变体:windowed 下 sys.stdout 是 None,任何输出都看不见;
#    console 变体是唯一的排障入口,也是 --tf-smoke 自检要看输出的那个。

param(
    [switch]$SkipFront,
    [ValidateSet('both', 'release', 'debug')]
    [string]$Only = 'both'
)

$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "找不到 uv。装法见 README「五分钟起步」,或 winget install --id=astral-sh.uv -e"
}

# ── 1. 前端产物 ────────────────────────────────────────────────────────────
Write-Host '[1/3] 前端产物' -ForegroundColor Cyan
if ($SkipFront) {
    Write-Host '  -SkipFront:跳过' -ForegroundColor DarkGray
} else {
    Push-Location (Join-Path $Project 'dev\front')
    try {
        if (-not (Test-Path 'node_modules')) { npm install }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw '前端构建失败' }
    } finally {
        Pop-Location
    }
}

# ── 2. 依赖同步 ────────────────────────────────────────────────────────────
# --frozen 是硬要求:pyproject 里 docling 写的是 >=2.128.0 **无上界**,
# 不锁 uv.lock 的话某次重跑会解析出完全不同的依赖树,而你会以为是 spec 写错了。
Write-Host '[2/3] 依赖同步(uv sync --frozen)' -ForegroundColor Cyan
$syncArgs = @('sync', '--extra', 'rag2', '--frozen')
& uv @syncArgs
if ($LASTEXITCODE -ne 0) { throw 'uv sync 失败' }

# ── 3. 打包 ────────────────────────────────────────────────────────────────
Write-Host '[3/3] 打包' -ForegroundColor Cyan
$Work = Join-Path $Project 'packaging\.pyinstaller'

$variants = @()
if ($Only -in @('both', 'release')) {
    $variants += @{ Name = 'TaskForce';       Console = '0'; Out = 'release' }
}
if ($Only -in @('both', 'debug')) {
    $variants += @{ Name = 'TaskForce-debug'; Console = '1'; Out = 'debug' }
}

foreach ($v in $variants) {
    $kind = if ($v.Console -eq '1') { 'console' } else { 'windowed' }
    Write-Host "  -> $($v.Name) [$kind]" -ForegroundColor Cyan
    $env:TF_CONSOLE = $v.Console
    $env:TF_NAME = $v.Name
    $distPath = Join-Path $Project "dist\$($v.Out)"
    $pyArgs = @(
        'run', '--frozen', 'pyinstaller', 'packaging/TaskForce.spec',
        '--noconfirm',
        '--workpath', $Work,
        '--distpath', $distPath
    )
    & uv @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "打包失败:$($v.Name)" }
}

Remove-Item Env:TF_CONSOLE, Env:TF_NAME -ErrorAction SilentlyContinue

Write-Host ''
Write-Host '完成:' -ForegroundColor Green
foreach ($v in $variants) {
    Write-Host "  dist\$($v.Out)\$($v.Name)\$($v.Name).exe"
}
Write-Host ''
Write-Host '自检(务必先跑这个,再双击):' -ForegroundColor Green
Write-Host '  dist\debug\TaskForce-debug\TaskForce-debug.exe --tf-smoke'

$ErrorActionPreference = "Stop"

# Optional maintainer helper. Runtime does not use repo-cache; bundled preview
# code lives in backend/app/preview/vendor. This script only refreshes upstream
# comparison copies when updating the vendored implementation.

$repos = @(
  @{ Name = "LiteVGGT-repo"; Url = "https://github.com/GarlicBa/LiteVGGT-repo.git"; Commit = "4767c17f8b6f176bb751566e92f60eb885040033" },
  @{ Name = "EDGS"; Url = "https://github.com/CompVis/EDGS.git"; Commit = "9a897645eb47c1b24d4f9e4428cd745927bf1ee1" },
  @{ Name = "lingbot-map"; Url = "https://github.com/Robbyant/lingbot-map.git"; Commit = "f720b421c6c50af3adc63272033226aa4811ef42" },
  @{ Name = "spark"; Url = "https://github.com/sparkjsdev/spark.git"; Commit = "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf" }
)

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$cache = Join-Path $root "repo-cache"
New-Item -ItemType Directory -Force -Path $cache | Out-Null

foreach ($repo in $repos) {
  $path = Join-Path $cache $repo.Name
  if (!(Test-Path $path)) {
    git clone $repo.Url $path
  }
  git -C $path fetch --all --tags
  git -C $path checkout $repo.Commit
  if ($repo.Name -eq "EDGS") {
    git -C $path submodule update --init --recursive submodules/gaussian-splatting
  }
}

Write-Host "Optional repository cache is ready at $cache"

后端服务启动命令
cd D:\Agent\Kyuriagents\libs\deepagents
# 如果还没加载 runtime.env，先加载
Get-Content .\deepagents\runtime\runtime.env | ForEach-Object {
  if ($_ -and -not $_.StartsWith("#")) {
    $name, $value = $_ -split "=", 2
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
  }
}

.\.venv\Scripts\python.exe scripts\api_server.py

前端服务启动命令

cd D:\Agent\Kyuriagents\apps\web
python -m http.server 5173 --bind 127.0.0.1

还要启动worker
.\.venv\Scripts\python.exe scripts\ingestion_worker.py
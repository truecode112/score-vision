module.exports = {
  apps: [
    {
      name: "sn44-miner",
      script: "main.py",
      interpreter: ".venv/bin/python",
      args: "-m uvicorn main:app --host 0.0.0.0 --port 7999",
      cwd: "./",
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "development",
      },
      env_production: {
        NODE_ENV: "production",
      },
    },
  ],
};

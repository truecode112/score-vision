module.exports = {
  apps: [
    {
      name: "sn44-validator",
      script: "main.py",
      interpreter: ".venv/bin/python",
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

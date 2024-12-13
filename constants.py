from datetime import timedelta

# Challenge intervals
REGULAR_CHALLENGE_INTERVAL = timedelta(seconds=20)
PREDICT_CHALLENGE_CHECK_INTERVAL = timedelta(minutes=1)

# Other constants
NETUID = 1  # Replace with your actual subnet ID
CHALLENGE_RESULT_TIMEOUT = timedelta(minutes=15)

CHALLENGE_INTERVAL=30
# MuZero


##Setup 

For first setup create a venv: `python -m venv venv`

Activate venv: `./venv/Scripts/activate`

To download requirements: `pip install -r requirements.txt`

If you need to add new requirements (for example numpy) you first pip install it as usual. Then you run `pip freeze > requirements.txt`. Make sure your venv is active for this step, if not you will add all python packages that you have globally to this file meaning everyone will download and update them frequently.  

To run tests the command is `python -m pytest`. All test names must start with "test" for pytest to reccognize it as a test funtion. For example `test_example_function`.

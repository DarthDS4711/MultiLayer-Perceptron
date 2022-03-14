def validate_integer(number):
    try:
        number = int(number)
        if number <= 0:
            return False
        else:
            return True
    except:
        return False

def validate_float(number):
    try:
        number = float(number)
        if number <= 0.0 or number > 1.0:
            return False
        else:
            return True
    except:
        return False
import pytest

@pytest.mark.parametrize("quality", [
    """ single sensor testing """
    (3, 'good'), 
    (3, 'moderate'),
    (3, 'bad'),
    
    (5, 'good'), 
    (5, 'moderate'),
    (5, 'bad'),
      
])

@pytest.fixture()
def compute_expected_accuracy(number_of_sensors, prob_correct):
      
    expected_accuracy = prob_correct ** number_of_sensors
    yield expected_accuracy
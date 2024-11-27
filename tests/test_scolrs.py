import unittest

from szkspyne.utils.scolrs import MeasureResult


json_str1 = '''
{
"measurements": {
    "CobbPT": {
    "Ok": -19.35
    },
    "CobbMT": {
    "Ok": 37.97
    },
    "CobbTLL": {
    "Ok": -28.25
    },
    "T1TiltAngle": {
    "Ok": -1.28
    },
    "CoronalBalance": {
    "Ok": 33.07
    },
    "ClavicleAngle": {
    "Ok": -0.163
    },
    "ShoulderHeight": {
    "Err": {
        "InvalidNumberOfPoints": {
        "IncorrectNumberOfPoints": [
            2,
            0
        ]
        }
    }
    },
    "PelvicObliquity": {
    "Ok": 5.48
    },
    "SacralObliquity": {
    "Ok": 1.39
    },
    "LegLengthDiscrepancy": {
    "Ok": -12.30
    }
},
"unit_of_length": "px"
}
'''

json_str2 = '''
{
"measurements": {
    "CobbPT": {
    "Ok": 52.07460590625324
    },
    "CobbMT": {
    "Ok": -39.09524866619732
    },
    "CobbTLL": {
    "Ok": 22.223657401334158
    },
    "T1TiltAngle": {
    "Ok": 16.83749767043411
    },
    "CoronalBalance": {
    "Ok": -14.63235000000003
    },
    "ClavicleAngle": {
    "Ok": 3.2875146554748698
    },
    "ShoulderHeight": {
    "Ok": -34.615369999999984
    },
    "PelvicObliquity": {
    "Ok": -4.564356694165012
    },
    "SacralObliquity": {
    "Ok": -2.1985693143716465
    },
    "LegLengthDiscrepancy": {
    "Ok": -16.153800000000047
    }
},
"unit_of_length": "px"
}
'''


class TestScolrs(unittest.TestCase):

    def test_measure_result(self):
        result = MeasureResult.model_validate_json(json_str1)
        self.assertIsInstance(result.measurements['CobbPT'], MeasureResult.Ok)
        self.assertEqual(result.measurements['CobbPT'].Ok, -19.35)
        self.assertIsInstance(result.measurements['ShoulderHeight'], MeasureResult.Error)
        self.assertEqual(
            result.measurements['ShoulderHeight'].Err, {'InvalidNumberOfPoints': {'IncorrectNumberOfPoints': [2, 0]}}
        )
        self.assertEqual(result.unit_of_length, 'px')

        oks = result.ok_measurements()
        self.assertEqual(len(oks), 9)
        errors = result.error_measurements()
        self.assertEqual(len(errors), 1)

    def test_differences(self):
        result1 = MeasureResult.model_validate_json(json_str1)
        result2 = MeasureResult.model_validate_json(json_str2)
        diff = result1.difference(result2)
        self.assertIsInstance(diff.measurements['CobbPT'], MeasureResult.Ok)
        self.assertAlmostEqual(diff.measurements['CobbPT'].Ok, -71.42460590625324)
        self.assertIsInstance(diff.measurements['ShoulderHeight'], MeasureResult.Error)
        self.assertEqual(diff.unit_of_length, 'px')

        diff = result2.difference(result1)
        self.assertAlmostEqual(diff.measurements['CobbPT'].Ok, 71.42460590625324)

        result2.unit_of_length = 'mm'
        with self.assertRaises(ValueError):
            diff = result1.difference(result2)

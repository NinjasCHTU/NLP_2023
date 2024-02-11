# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 08:51:01 2023

@author: Heng2020
"""
import unittest
from try01 import St_SplitSentence

class TestSt_SplitSentence(unittest.TestCase):

    def test_basic_case_inplace_true(self):
        lst01 = ["This is cool", "-Apple is great. -But Grape is better", "penis"]
        expected = ['This is cool', 'Apple is great.', 'But Grape is better', 'penis']
        St_SplitSentence(lst01, "-")
        self.assertEqual(lst01, expected)
        
    def test_empty_list_inplace_true(self):
        lst02 = []
        expected = []
        St_SplitSentence(lst02, "-", inplace=True)
        self.assertEqual(lst02, expected)
        
    def test_no_delimiter_inplace_true(self):
        lst03 = ["Hello", "World"]
        expected = ["Hello", "World"]
        St_SplitSentence(lst03, "-", inplace=True)
        self.assertEqual(lst03, expected)
        
    def test_only_delimiters_inplace_true(self):
        lst04 = ["-", "--", "---"]
        expected = []
        St_SplitSentence(lst04, "-", inplace=True)
        self.assertEqual(lst04, expected)
        
    def test_trailing_spaces_inplace_true(self):
        lst05 = ["  -Apple is great. -  But Grape is better  ", "penis  "]
        expected = ['Apple is great.', 'But Grape is better', 'penis']
        St_SplitSentence(lst05, "-", inplace=True)
        self.assertEqual(lst05, expected)
        
    def test_basic_case_inplace_false(self):
        lst01 = ["This is cool", "-Apple is great. -But Grape is better", "penis"]
        expected = ['This is cool', 'Apple is great.', 'But Grape is better', 'penis']
        result = St_SplitSentence(lst01, "-", inplace=False)
        self.assertEqual(result, expected)
        self.assertNotEqual(lst01, result) 
        

unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestSt_SplitSentence))
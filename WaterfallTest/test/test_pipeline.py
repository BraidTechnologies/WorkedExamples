# Copyright (c) 2024 Braid Technologies Ltd

# Standard Library Imports
import pytest
import os
import sys
import logging

from src.search_pipeline import WaterfallDataPipeline

test_root = os.path.dirname(__file__)
parent= os.path.abspath(os.path.join(test_root, '..'))
src_dir = os.path.join(parent, 'src')
sys.path.extend([parent, src_dir])

# Set up logging to display information about the execution of the script
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

def test_basic ():
    test_output_location = 'test_output'
    pipeline = WaterfallDataPipeline (test_output_location)
    assert pipeline.output_location == test_output_location 

@pytest.mark.timeout(2000)
def test_with_search ():
    test_root = os.path.dirname(__file__)
    os.chdir (test_root)
    test_output_location = 'test_output'

    pipeline = WaterfallDataPipeline (test_output_location)
    links = pipeline.search (7)    
    assert len(links) >= 1   


# multimedia-2023

This repository contains files from the 2023 project in the Multimedia Systems course of Aristotle University of Thessaloniki. The subject of this project is creating a simplified MP3 codec. 

**Sub-band filtering** \
Creates a buffer and using analysis and synthesis filters can deconstruct and reconstruct the signal in sub-bands. 
 
The frequency response of the analysis filters can be seen below:

<img width="497" alt="image" src="https://user-images.githubusercontent.com/95578892/233616919-788d1bd3-a4c1-4e74-9ef2-23fad4bba1c0.png">

**DCT** \
Calculated the DCT and inverse DCT of the input signal

**Calculating the hearing threshold** \
Calculates the hearing threshold in every frame using the psychoacoustic model. For example, the hearing thresholds for the 3rd and 7th frame can be seen below:
<div id="image-table">
    <table>
	    <tr>
          <td style="padding:10px">
            	<img src="https://user-images.githubusercontent.com/95578892/233618462-317783f8-1c6b-45c7-8f8d-fd2a85a8c47a.png" width="350"/>
          </td>
    	    <td style="padding:10px">
        	    <img src="https://user-images.githubusercontent.com/95578892/233618517-4ec829d6-bc0a-46fb-9341-ecf6ae0778ff.png" width="350"/>
      	    </td>
        </tr>
    </table>
</div>

**Quantization and Dequantization** \
Implements the quantization and dequantization function using the psycoacoustic model and the hearing threshold calculated prior. 

**Run-length Encoding** \
Implements the RLE routine and returns the coded and decoded signal.

**Huffman coding** \
Returns the betstream of the coded signal using the RLE calculated prior

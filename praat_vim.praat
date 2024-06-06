form args
#	sentence filedir
#	word fileroot
	sentence filepath
	real time_start
	real time_end
	integer spect_min
	integer spect_max
	real silence_threshold
	real voicing_threshold
	real octave_cost
	real octave_jump_cost
	real voiced_unvoiced_cost
	integer hann
	real denoise
endform
# sound = Read from file: filedir$ + fileroot$ + ".wav"
sound = Read from file: filepath$
select sound

# abrir filename$_band con pitch trace y hann band opcional

if hann = 1 and denoise = 0
	Filter (pass Hann band)... spect_min spect_max 100
	# luego de hacer el filtering te queda seleccionado en el object window
	sound_band = selected ()
	select sound_band
	View & Edit
	editor: sound_band
		Select: time_start,time_end
		Spectrogram settings... spect_min spect_max 0.01 70
		Pitch settings... spect_min spect_max Hertz autocorrelation automatic
		Advanced pitch settings... 0 0 no 15 silence_threshold voicing_threshold octave_cost octave_jump_cost voiced_unvoiced_cost
	endeditor
endif
if hann = 1 and denoise <> 0
	Reduce noise... 0.0 denoise 0.025 80.0 10000.0 40.0 -20.0 spectral-subtraction
	# luego de hacer el denoising te queda seleccionado en el object window
	sound_band = selected ()
	select sound_band
	Filter (pass Hann band)... spect_min spect_max 100
	sound_band = selected ()
	select sound_band
	View & Edit
	editor: sound_band
		Select: time_start,time_end
		Spectrogram settings... spect_min spect_max 0.01 70
		Pitch settings... spect_min spect_max Hertz autocorrelation automatic
		Advanced pitch settings... 0 0 no 15 silence_threshold voicing_threshold octave_cost octave_jump_cost voiced_unvoiced_cost
	endeditor
endif
if hann = 0 and denoise <> 0
	Reduce noise... 0.0 denoise 0.025 80.0 10000.0 40.0 -20.0 spectral-subtraction
	sound = selected ()
	select sound
	# luego de hacer el denoising te queda seleccionado en el object window
	View & Edit
	editor: sound
		Select: time_start,time_end
		Spectrogram settings... spect_min spect_max 0.01 70
		Pitch settings... spect_min spect_max Hertz autocorrelation automatic
		Advanced pitch settings... 0 0 no 15 silence_threshold voicing_threshold octave_cost octave_jump_cost voiced_unvoiced_cost
	endeditor
endif
if hann = 0 and denoise = 0
	View & Edit
	editor: sound
		Select: time_start,time_end
		Spectrogram settings... spect_min spect_max 0.01 70
		Pitch settings... spect_min spect_max Hertz autocorrelation automatic
		Advanced pitch settings... 0 0 no 15 silence_threshold voicing_threshold octave_cost octave_jump_cost voiced_unvoiced_cost
	endeditor
endif

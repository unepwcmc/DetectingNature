require_relative 'DetectingNature'

settings = DetectingNature::SettingsManager.new('settings-fast.json')
framework = DetectingNature::ClassificationFramework.new('data/SSUN-2', settings, false)
framework.train

results = framework.classify('data/Testing/natural')

results.each do |result|
	puts result.filepath, result.category, result.certainty, '-' * 20
end

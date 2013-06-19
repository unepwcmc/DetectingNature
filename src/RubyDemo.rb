require_relative 'DetectingNature'

settings = DetectingNature::SettingsManager.new('settings-fast.json')
framework = DetectingNature::ClassificationFramework.new('data/SSUN-2', settings, false)
results = framework.classify('data/Testing/natural')

results.each { |result|
	puts result.filepath, result.category, result.certainty, '-' * 20
}

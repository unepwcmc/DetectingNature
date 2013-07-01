Gem::Specification.new do |s|
	s.name = 'DetectingNature'
	s.version = '1.0.0'

	s.summary = 'Framework for automated image classification'
	s.description = 'Flexible framework that provides a diverse set of '\
		'techniques for automated, multi-class image classification'
	s.homepage = 'https://github.com/unepwcmc/DetectingNature'

	s.author = 'Tiago Andrade'
	s.email = 'tiago.andrade@unep-wcmc.org'
	
	s.files = `git ls-files`.split("\n")
	s.require_path = 'ext'
	s.extensions << 'extconf.rb'
end

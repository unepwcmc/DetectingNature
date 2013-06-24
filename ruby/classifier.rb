require 'open-uri'
require 'fileutils'

require_relative 'apikeys'
require_relative 'DetectingNature'

class Classifier
	include DetectingNature
	
	def process_image(id, url)
		filename = download_image(id, url)
		category, certainty = classify_file(filename)
		return filename, category, certainty
	end
		
	def move_file(filename, category, certainty)
		certainty_trunc = "%0.6f" % certainty
		dst = @base_folder + "results/#{category}/#{certainty_trunc}.jpg"
		FileUtils.mkdir_p(File.dirname(dst))
		FileUtils.cp(filename, dst)
	end
	
	def save_cartodb(lat, lon, classification, probability, picture_date, url)	
		`curl --data "api_key=#{ApiKeys::CARTODB_KEY}&q=INSERT INTO picture_classifier (the_geom, classification, picture_date, probability, url) VALUES (
	ST_PointFromText('POINT(#{lon} #{lat})', 4326), '#{classification}', to_date('#{picture_date}', 'yyyy-mm-dd'), #{probability}, '#{url}')" http://carbon-tool.cartodb.com/api/v2/sql`
	end

	protected

	def initialize(base_folder)
		@base_folder = base_folder
		@settings = SettingsManager.new('settings-ruby.json')
		@framework = ClassificationFramework.new('data/SSUN-2', @settings, false)
		@framework.train
	end
	
	def download_image(id, url)
		filename = @base_folder + "#{id}.jpg"
		
		# Do not download an image again if it already exists
		return filename if File.file? filename
		
		open(url, 'rb') do |read_file|
			# Ensure the folder exists before trying to save the file there
			FileUtils.mkdir_p(File.dirname(filename))
			File.open(filename, 'wb') do |save_file|
				save_file.write(read_file.read)
			end
		end
		
		return filename
	end

	def classify_file(filename)
		results = @framework.classify(filename).first
		return results.category, results.certainty
	end
end

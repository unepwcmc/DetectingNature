#!/usr/bin/env ruby

require 'csv'

require_relative 'classifier'

class PanoramioCsvClassifier < Classifier
	def initialize
		super 'data/panoramio/'
	end

	def process_csv
		CSV.foreach('panoramio_data.csv', :headers => true) do |row|
			begin
				filename, category, certainty = 
					process_image(row['id'], row['image_url'])
				
				puts filename, row['Lat'], row['Lng'], category, certainty, '-' * 20
		
				move_file filename, category, certainty
			rescue Exception => e
				puts "Image #{row['id']} not available (#{e.message})", '-' * 20
			end
		end
	end
end

panoramiocsv_classifier = PanoramioCsvClassifier.new
panoramiocsv_classifier.process_csv

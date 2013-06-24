#!/usr/bin/env ruby

require 'date'
require 'panoramio-rb'

require_relative 'classifier'

class PanoramioClassifier < Classifier
	def initialize
		super 'data/panoramio/'
	end
		
	def search_panoramio(start)
		panoramio = PanoramioRb.get_panoramas(
			:set => :full,
			:mapfilter => false,
			:from => start,
			:to => start + 100,
			:minx => '-180.0',
			:miny => '-90.0',
			:maxx => '180.0',
			:maxy => '90.0')
	
		puts "Total of #{panoramio[:count]} images (downloading #{start}-#{start+100})", '-' * 20
		return panoramio
	end

	def process_area
		from = 0
		begin
			panoramio = search_panoramio from
			panoramio.photos.each do |photo|
				begin
					filename, category, certainty = 
						process_image(photo.photo_id, photo.photo_file_url)
				
					puts filename, photo.latitude, photo.longitude,
						Date.parse(photo.upload_date).to_s,
						category, certainty, '-' * 20
		
					save_cartodb(photo.latitude, photo.longitude, category,
						certainty, Date.parse(photo.upload_date).to_s,
						photo.photo_file_url)
					#move_file filename, category, certainty
				rescue Exception => e
					puts "Image #{photo.photo_id} not available (#{e.message})", '-' * 20
				end
			end
			
			from += 100
		end while panoramio.has_more
	end
end

panoramio_classifier = PanoramioClassifier.new
panoramio_classifier.process_area

#!/usr/bin/env ruby

require 'flickraw'
require 'time'

require_relative 'classifier'

class FlickrClassifier < Classifier
	def initialize
		super 'data/flickr/'
		FlickRaw.api_key = ApiKeys::FLICKR_KEY
		FlickRaw.shared_secret = ApiKeys::FLICKR_SECRET
	end
	
	def search_flickr(min_date, page)
		max_date = min_date + 600

		return flickr.photos.search(
			:min_upload_date => min_date,
			:max_upload_date => max_date, # minimum of 10min interval
			#:min_taken_date => min_date,
			#:max_taken_date => max_date,
			#:text => "-instagramapp",
			:content_type => 1, # photos only
			:media => 'photos', # no videos
			:accuracy => 16, # street-level accuracy
			#:geo_context => 2, # outdoors
			:has_geo => 1,
			:extras => 'date_upload,date_taken,geo,tags',
			:per_page => 250,
			:page => page)
	end

	def process_list(imagelist)
		imagelist.each do |image|
			begin		
				image_url = FlickRaw.url(image)
			
				puts image.id, image.latitude, image.longitude, image.accuracy, image.tags,
					Time.at(image.dateupload.to_i).to_s, image.datetaken, image_url
	
				filename, category, certainty = 
					process_image(image.id, image_url)
				puts filename, category, certainty, '-' * 20
				
				save_cartodb(image.latitude, image.longitude, category,
					certainty, Time.parse(image.datetaken).to_date.to_s,
					image_url)
			
				#move_file filename, category, certainty
			rescue Exception => e
				puts "Image not available (#{e.message})", '-' * 20
			end
		end
	end

	def process_dates(current_date, end_date)
		while current_date < end_date do
			puts "Starting with #{current_date}"
		
			current_page = 0
			begin
				current_page += 1
				imagelist = search_flickr current_date.to_i, current_page
				puts "Downloading page #{current_page}/#{imagelist.pages} (#{imagelist.total} images)", '-' * 20
				process_list imagelist
			end while current_page < imagelist.pages
			
			current_date += 600
		end
	end
end

if ARGV.size != 2 then
	puts 'Please provide the start and end date/time as arguments'
else
	flickr_classifier = FlickrClassifier.new
	flickr_classifier.process_dates Time.parse(ARGV[0]), Time.parse(ARGV[1])
end


#!/usr/bin/env ruby

require 'flickraw'
require 'thread/pool'
require 'time'

require_relative 'classifier'

class FlickrClassifier < Classifier
	@@total_threads = 1
	@@stopping = false

	def initialize
		super 'data/flickr/'
		FlickRaw.api_key = ApiKeys::FLICKR_KEY
		FlickRaw.shared_secret = ApiKeys::FLICKR_SECRET
		
		Signal.trap('USR1') do
			@@total_threads += 1
			puts "Increasing thread pool size to #{@@total_threads}"
		end
		
		Signal.trap('USR2') do
			@@total_threads -= 1
			puts "Decreasing thread pool size to #{@@total_threads}"
		end
		
		Signal.trap('INT') do
			@@stopping = true
			puts 'Stopping classifier when current image batch is done'
		end
	end
	
	def search_flickr(min_date, page)
		# Flickr upload date search seems to have a resolution of 10 minutes
		# Use a multiple of 10 minutes to prevent repeated or missing images
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

	def process_one(image)
		begin		
			image_url = FlickRaw.url(image)
	
			filename, category, certainty = 
				process_image(image.id, image_url)
	
			puts image.id, image.latitude, image.longitude, image.accuracy,
				image.tags, Time.at(image.dateupload.to_i).to_s,
				image.datetaken, image_url, filename, category,
				certainty, '-' * 20
		
			save_cartodb(image.latitude, image.longitude, category,
				certainty, Time.parse(image.datetaken).to_date.to_s,
				image_url)
	
			#move_file filename, category, certainty
		rescue Exception => e
			puts "Image not available (#{e.message})", '-' * 20
		end
	end

	def process_list(imagelist)
		pool = Thread.pool(@@total_threads)
		
		imagelist.each do |thread_image|
			pool.process(thread_image) { |image|
				process_one image
			}
		end
		
		pool.shutdown
	end

	def process_dates(current_date, end_date)
		while current_date < end_date do
			puts "Starting with #{current_date}"
		
			begin
				current_page = 0
				begin
					current_page += 1
					imagelist = search_flickr current_date.to_i, current_page
					puts "Downloading page #{current_page}/#{imagelist.pages}"\
						"(#{imagelist.total} images)", '-' * 20
					if imagelist.pages.to_i <= 1 or imagelist.total.to_i > 0 then
						process_list imagelist
					else
						current_page -= 1
						sleep 30
					end
				end while current_page < imagelist.pages
			rescue Exception => e
				puts "Image list retrieval failed (#{e.message})", '-' * 20
			end
			
			current_date += 600			
			
			if @@stopping
				puts "Stopping at #{current_date}"
				exit
			end
		end
	end
end

if ARGV.size != 2 then
	puts 'Please provide the start and end date/time as arguments'
else
	flickr_classifier = FlickrClassifier.new
	flickr_classifier.process_dates Time.parse(ARGV[0]), Time.parse(ARGV[1])
end


#!/usr/bin/env ruby

require 'tweetstream'

require_relative 'classifier'

class TwitterStreamClassifier < Classifier
	def initialize
		super 'data/twitter/'
		TweetStream.configure do |config|
			config.consumer_key = ApiKeys::TWITTER_KEY
			config.consumer_secret = ApiKeys::TWITTER_SECRET
			config.oauth_token = ApiKeys::TWITTER_OAUTH_TOKEN
			config.oauth_token_secret = ApiKeys::TWITTER_OAUTH_SECRET
			config.auth_method = :oauth
		end
	end
	
	def process_recent
		TweetStream::Client.new.track('photo', 'picture', 'photograph') do |status|
			begin
				next if status.media.empty? or (status.geo.nil? and status.place.nil?)
				
				filename, category, certainty = 
					process_image(status.media.first.id, status.media.first.media_url)
				
				puts status.from_user
				puts "#{status.geo.lat}, #{status.geo.lng}" unless status.geo.nil?
				puts status.place.bounding_box.coordinates.join(', ') unless status.place.nil?
				puts status.media.first.media_url
				puts status.text
				puts filename, category, certainty, '-' * 20
				
				move_file filename, category, certainty
			rescue Exception => e
				puts "Image not available (#{e.message})", '-' * 20
			end
		end
	end
end

twitterstream_classifier = TwitterStreamClassifier.new
twitterstream_classifier.process_recent

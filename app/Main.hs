{-# LANGUAGE RecordWildCards #-}


module Main 
  ( main
  ) where


import           Control.Monad (forM_)
import           Data.Monoid ((<>))
import           Options.Applicative
-- import           Data.Maybe (mapMaybe)
import           Data.Ord (comparing)
import           Data.String (fromString)
import           Data.List (sortBy)
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as TL

import qualified Dhall as Dhall

-- import           System.FilePath (isAbsolute, (</>))

-- import           System.Environment (getArgs)

-- import qualified SMT as SMT
-- import qualified Net.Ex1 as Ex1
-- import qualified Net.Ex2 as Ex2
-- import qualified Net.Ex3 as Ex3
-- import qualified Net.Ex4 as Ex4
-- import qualified Net.DAG as DAG
import qualified Net.ArcGraph as Graph
import qualified Net.POS as POS
-- import qualified Net.MWE2 as MWE
import qualified MWE
import qualified GradientDescent.Momentum as Mom
import qualified Embedding.Dict as D
import qualified Embedding as Emb
import qualified Format.Cupt as Cupt


--------------------------------------------------
-- Commands
--------------------------------------------------


-- | Available commands
data Command
    = FastText Emb.Config
      -- ^ Calculate fasttext embeddings for the individual words in the file
    | Train TrainConfig
      -- ^ Train a model
    | Tag 
      { inpCupt  :: FilePath
      , inpEmbs  :: FilePath
      , inpModel :: FilePath
      , mweTyp   :: T.Text
      }
      -- ^ Tagging


-- | Training configuration
data TrainConfig = TrainConfig
  { trainCupt :: FilePath
  , trainEmbs :: FilePath
  , trainTmpDir :: FilePath
    -- ^ For temporary storage
  , trainMweCat :: T.Text
    -- ^ MWE category (e.g., LVC) to focus on
  , trainInModel   :: Maybe FilePath
    -- ^ Input model (otherwise, random)
  , trainOutModel  :: Maybe FilePath
    -- ^ Where to store the output model
  , trainCfgPath :: Maybe FilePath
    -- ^ Additional training configuration path
  }


--------------------------------------------------
-- Parse options
--------------------------------------------------


fastTextOptions :: Parser Command
fastTextOptions = fmap FastText $ Emb.Config
  <$> strOption
        ( metavar "FILE"
       <> long "embed"
       <> short 'e'
       <> help "Fasttext embedding file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "input"
       <> short 'i'
       <> help "Input .cupt file"
        )
  <*> switch
        ( long "no-header"
       <> short 'n'
       <> help "Embedding file has no header"
        )


trainOptions :: Parser Command
trainOptions = fmap Train $ TrainConfig
  <$> strOption
        ( metavar "FILE"
       <> long "input"
       <> short 'i'
       <> help "Input .cupt file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "embed"
       <> short 'e'
       <> help "Input embedding file"
        )
  <*> strOption
        ( metavar "DIR"
       <> long "tmp"
       <> help "Directory to store temporary stuff"
        )
  <*> strOption
        ( long "mwe"
       <> short 'm'
       <> help "MWE category to focus on"
        )
  <*> (optional . strOption)
        ( long "in-model"
       <> help "Input model"
        )
  <*> (optional . strOption)
        ( metavar "FILE"
       <> long "output"
       <> short 'o'
       <> help "Output model"
        )
  <*> (optional . strOption)
        ( metavar "FILE"
       <> long "config"
       <> short 'c'
       <> help "Additional training configuration"
        )


tagOptions :: Parser Command
tagOptions = Tag
  <$> strOption
        ( metavar "FILE"
       <> long "input"
       <> short 'i'
       <> help "Input .cupt file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "embed"
       <> short 'e'
       <> help "Input embedding file"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "params"
       <> short 'p'
       <> help "Input model parameters file"
        )
  <*> strOption
        ( long "mwe"
       <> short 'm'
       <> help "MWE category to focus on"
        )


--------------------------------------------------
-- Global options
--------------------------------------------------


opts :: Parser Command
opts = subparser
  ( command "fasttext"
    (info (helper <*> fastTextOptions)
      (progDesc "Determine fasttext embeddings")
    )
    <> command "train"
    (info (helper <*> trainOptions)
      (progDesc "Training")
    )
    <> command "tag"
    (info (helper <*> tagOptions)
      (progDesc "Tagging")
    )
  )


--------------------------------------------------
-- Main
--------------------------------------------------


-- | Run program depending on the cmdline arguments.
run :: Command -> IO ()
run cmd =
  case cmd of

    FastText cfg -> do
      Emb.produceEmbeddings cfg

    Train TrainConfig{..} -> do

--       let configPath' =
--             if isAbsolute configPath
--             then configPath
--             else "./" </> configPath
--           tagConfig = Task.TagConfig
--             { tagBestPath = bestPath
--             , splitMwesOn = splitOn }
      config <- 
        case trainCfgPath of
          Nothing -> return MWE.defTrainCfg
          Just configPath -> Dhall.detailed
            (Dhall.input Dhall.auto $ fromString configPath)

      -- Initial network
      net0 <- Graph.new 300 2
      -- Read .cupt (ignore paragraph boundaries); need to apply
      -- `tail.decorate` because we don't want the dummy root!
      cupt <- map (tail . Cupt.decorate) . concat
        <$> Cupt.readCupt trainCupt
      -- Read the corresponding embeddings
      embs <- Emb.readEmbeddings trainEmbs
      net <- MWE.train config trainTmpDir trainMweCat
        ( map
            (uncurry MWE.Sent)
            (zip cupt embs)
        ) net0
      case trainOutModel of
        Nothing -> return ()
        Just path -> do
          putStrLn "Saving model..."
          Graph.saveParam path net
      putStrLn "Done!"

    Tag{..} -> do
      -- Load the model
      net <- Graph.loadParam inpModel
      -- Read .cupt (ignore paragraph boundaries); need to apply
      -- `tail.decorate` because we don't want the dummy root!
      cupt <- map (tail . Cupt.decorate) . concat
        <$> Cupt.readCupt inpCupt
      -- Read the corresponding embeddings
      embs <- Emb.readEmbeddings inpEmbs
      MWE.tagMany net mweTyp
        ( map
            (uncurry MWE.Sent)
            (zip cupt embs)
        )


main :: IO ()
main =
    execParser optsExt >>= run
  where
    optsExt = info (helper <*> opts)
       ( fullDesc
      <> progDesc "MWE identification tool"
      <> header "galago" )

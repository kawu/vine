{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}


module Main 
  ( main
  ) where


import           GHC.TypeNats (KnownNat)

import           Control.Monad (forM_, guard)

import           Options.Applicative
import           Data.Monoid ((<>))
import           Data.Maybe (mapMaybe)
import           Data.Ord (comparing)
import           Data.String (fromString)
import           Data.List (sortBy)
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as TL

import qualified Numeric.LinearAlgebra.Static as LA

import qualified Dhall as Dhall

import           System.FilePath (isAbsolute, (</>))

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
-- import qualified GradientDescent.Momentum as Mom
import qualified Embedding.Dict as D
import qualified Embedding as Emb
import qualified Format.Cupt as Cupt

import Debug.Trace (trace)


--------------------------------------------------
-- Commands
--------------------------------------------------


-- | Available commands
data Command
    = FastText Emb.Config
      -- ^ Calculate fasttext embeddings for the individual words in the file
    | Train TrainConfig
      -- ^ Train a model
    | Tag TagConfig
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


-- | Tagging configuration
data TagConfig = TagConfig
  { tagCupt :: FilePath
  , tagEmbs :: FilePath
  , tagMweCat :: T.Text
    -- ^ MWE category (e.g., LVC) to focus on
  , tagModel   :: FilePath
    -- ^ Input model (otherwise, random)
  , tagDepth   :: Int
    -- ^ Input model (otherwise, random)
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
       <> short 't'
       <> help "MWE category (type) to learn"
        )
  <*> (optional . strOption)
        ( metavar "FILE"
       <> long "model"
       <> short 'm'
       <> help "Input model"
        )
  <*> (optional . strOption)
        ( metavar "FILE"
       <> long "out-model"
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
tagOptions = fmap Tag $ TagConfig
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
        ( long "mwe"
       <> short 't'
       <> help "MWE category (type)"
        )
  <*> strOption
        ( metavar "FILE"
       <> long "model"
       <> short 'm'
       <> help "Input model"
        )
  <*> option auto
        ( long "depth"
       <> short 'd'
       <> help "Recursion depth"
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

      -- Training configuration
      config <- 
        case trainCfgPath of
          Nothing -> return MWE.defTrainCfg
          Just configPath ->
            Dhall.input Dhall.auto (dhallPath configPath)

      -- Initial network
      net0 <- 
        case trainInModel of
          Nothing -> Graph.new 300 2
          Just path -> Graph.loadParam path
      -- Read .cupt (ignore paragraph boundaries)
      cupt <- map Cupt.decorate . concat
        <$> Cupt.readCupt trainCupt
      -- Read the corresponding embeddings
      embs <- Emb.readEmbeddings trainEmbs
      net <- MWE.train config trainTmpDir trainMweCat
        (mkInput cupt embs) net0
      case trainOutModel of
        Nothing -> return ()
        Just path -> do
          putStrLn "Saving model..."
          Graph.saveParam path net
      putStrLn "Done!"

    Tag TagConfig{..} -> do
      -- Load the model
      net <- Graph.loadParam tagModel
      -- Read .cupt (ignore paragraph boundaries)
      cupt <- map Cupt.decorate . concat
        <$> Cupt.readCupt tagCupt
      -- Read the corresponding embeddings
      embs <- Emb.readEmbeddings tagEmbs
      MWE.tagMany tagMweCat net tagDepth
        (mkInput cupt embs)


main :: IO ()
main =
    execParser optsExt >>= run
  where
    optsExt = info (helper <*> opts)
       ( fullDesc
      <> progDesc "MWE identification tool"
      <> header "galago" )


--------------------------------------------------
-- Utils
--------------------------------------------------


mkInput
  :: (KnownNat d)
  => [Cupt.Sent]
  -> [Maybe [LA.R d]]
  -> [MWE.Sent d]
mkInput cupt embs =
  ( map
      (uncurry MWE.Sent)
      (mapMaybe clear $ zip cupt embs)
  ) 
  where
    clear (sent, mayEmbs) = report sent $ do
      embs <- mayEmbs
      guard $ simpleSent sent
      return (sent, embs)
    report sent mayVal = 
      case mayVal of
        Just val -> Just val
        Nothing -> 
          let sentTxt = T.intercalate " " (map Cupt.orth sent)
           in trace ("Ignoring sentence: " ++ T.unpack sentTxt) Nothing
    simpleSent sent = and $ do
      tok <- sent
      return $ case Cupt.tokID tok of
                 Cupt.TokID _ -> True
                 _ -> False


dhallPath :: FilePath -> Dhall.Text
dhallPath path = fromString $
  if isAbsolute path
  then path
  else "./" </> path
